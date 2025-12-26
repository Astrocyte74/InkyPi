from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timedelta
from contextlib import nullcontext

from flask import Blueprint, jsonify, request, current_app

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _require_token() -> bool:
    """Require a token only if configured via INKYPI_SHORTCUTS_TOKEN."""
    device_config = current_app.config["DEVICE_CONFIG"]
    expected = (device_config.load_env_key("INKYPI_SHORTCUTS_TOKEN") or "").strip()
    if not expected:
        return True

    provided = (request.headers.get("X-InkyPi-Token") or request.args.get("token") or "").strip()
    return provided == expected


def _fmt_dt(dt: datetime) -> str:
    try:
        return dt.strftime("%b %d %I:%M %p").replace(" 0", " ")
    except Exception:
        return dt.isoformat()


def _human_ago(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds < 120:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _format_in(seconds: int | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"in {seconds}s"
    if seconds < 3600:
        return f"in {seconds // 60}m {seconds % 60}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"in {hours}h {minutes}m"


def _display_title(plugin_id: str | None, plugin_instance: str | None) -> str:
    pid = (plugin_id or "").strip().lower()
    if pid == "daily_cat_weather":
        base = "🎨 Illustration"
    elif pid == "daily_theme_card":
        base = "🗒️ Daily Card"
    elif pid == "world_flags":
        base = "🏳️ World Flags"
    elif pid:
        base = pid
    else:
        base = "Status"

    inst = (plugin_instance or "").strip()
    return f"{base} ({inst})" if inst else base


def _build_status_payload(*, now: datetime, include_image: bool, include_image_base64: bool) -> dict:
    device_config = current_app.config["DEVICE_CONFIG"]

    refresh_info = None
    try:
        refresh_info = device_config.get_refresh_info()
    except Exception:
        refresh_info = None

    plugin_id = getattr(refresh_info, "plugin_id", None) if refresh_info else None
    plugin_instance = getattr(refresh_info, "plugin_instance", None) if refresh_info else None
    refresh_time_raw = getattr(refresh_info, "refresh_time", None) if refresh_info else None

    last_refresh_iso = None
    last_refresh_human = None
    last_refresh_ago = None
    next_refresh_iso = None
    next_refresh_human = None
    seconds_until_next = None

    try:
        if refresh_time_raw:
            last_dt = datetime.fromisoformat(str(refresh_time_raw))
            if last_dt.tzinfo is None and now.tzinfo is not None:
                last_dt = last_dt.replace(tzinfo=now.tzinfo)
            last_refresh_iso = last_dt.isoformat()
            last_refresh_human = _fmt_dt(last_dt)
            last_refresh_ago = _human_ago(int((now - last_dt).total_seconds()))

            cycle = int(device_config.get_config("plugin_cycle_interval_seconds", default=0) or 0)
            if cycle > 0:
                next_dt = last_dt + timedelta(seconds=cycle)
                next_refresh_iso = next_dt.isoformat()
                next_refresh_human = _fmt_dt(next_dt)
                seconds_until_next = max(0, int((next_dt - now).total_seconds()))
    except Exception:
        logger.exception("Failed computing refresh timing.")

    image_b64 = None
    image_bytes = None
    current_path = getattr(device_config, "current_image_file", None)
    if include_image and current_path and os.path.exists(current_path):
        try:
            with open(current_path, "rb") as handle:
                image_bytes = handle.read()
            if include_image_base64:
                image_b64 = base64.b64encode(image_bytes).decode("ascii")
        except Exception:
            logger.exception("Failed reading current image.")

    payload = {
        "ok": True,
        "status_text": "\n".join(
            [
                _display_title(plugin_id, plugin_instance),
                f"Last refresh: {last_refresh_human} ({last_refresh_ago})" if last_refresh_human else "Last refresh: unknown",
                (
                    f"Next refresh: {_format_in(seconds_until_next)}"
                    if seconds_until_next is not None
                    else "Next refresh: unknown"
                ),
            ]
        ),
        "plugin_id": plugin_id,
        "plugin_instance": plugin_instance,
        "last_refresh": {
            "iso": last_refresh_iso,
            "human": last_refresh_human,
            "ago": last_refresh_ago,
        },
        "next_refresh": {
            "iso": next_refresh_iso,
            "human": next_refresh_human,
            "in_seconds": seconds_until_next,
        },
        "image": {
            "present": bool(image_bytes),
            "content_type": "image/png" if image_bytes else None,
            "base64_png": image_b64,
        },
    }
    return payload


@api_bp.route("/status", methods=["GET"])
def status():
    """
    iOS Shortcuts-friendly status endpoint.

    Returns JSON including a base64-encoded PNG of the current display (optional).
    """
    if not _require_token():
        return jsonify({"error": "Unauthorized"}), 401

    include_image = str(request.args.get("image", "1")).strip().lower() not in {"0", "false", "no"}
    include_image_base64 = str(request.args.get("format", "base64")).strip().lower() in {"base64", "b64", "json"}

    now = None
    try:
        refresh_task = current_app.config.get("REFRESH_TASK")
        now = refresh_task._get_current_datetime() if refresh_task and hasattr(refresh_task, "_get_current_datetime") else None
    except Exception:
        now = None
    if now is None:
        now = datetime.utcnow()

    return jsonify(_build_status_payload(now=now, include_image=include_image, include_image_base64=include_image_base64))


@api_bp.route("/next", methods=["GET", "POST"])
def next_item():
    """Advance the display to the next playlist item (iOS Shortcuts-friendly)."""
    if not _require_token():
        return jsonify({"error": "Unauthorized"}), 401

    include_image = str(request.args.get("image", "1")).strip().lower() not in {"0", "false", "no"}
    include_image_base64 = str(request.args.get("format", "base64")).strip().lower() in {"base64", "b64", "json"}
    force_refresh = str(request.args.get("force", "0")).strip().lower() in {"1", "true", "yes"}

    device_config = current_app.config["DEVICE_CONFIG"]
    display_manager = current_app.config["DISPLAY_MANAGER"]
    refresh_task = current_app.config.get("REFRESH_TASK")

    now = None
    try:
        now = refresh_task._get_current_datetime() if refresh_task and hasattr(refresh_task, "_get_current_datetime") else None
    except Exception:
        now = None
    if now is None:
        now = datetime.utcnow()

    playlist_manager = device_config.get_playlist_manager()
    playlist = playlist_manager.determine_active_playlist(now)
    if not playlist or not playlist.plugins:
        return jsonify({"error": "No active playlist with plugins."}), 400

    lock = getattr(refresh_task, "lock", None)
    ctx = lock if lock is not None else nullcontext()

    from plugins.plugin_registry import get_plugin_instance
    from refresh_task import PlaylistRefresh
    from utils.image_utils import compute_image_hash
    from model import RefreshInfo

    with ctx:
        plugin_instance = playlist.get_next_plugin()
        plugin_config = device_config.get_plugin(plugin_instance.plugin_id)
        if not plugin_config:
            return jsonify({"error": f"Plugin '{plugin_instance.plugin_id}' not found."}), 404

        plugin = get_plugin_instance(plugin_config)
        image = PlaylistRefresh(playlist, plugin_instance, force=force_refresh).execute(plugin, device_config, now)

        display_manager.display_image(image, image_settings=plugin_config.get("image_settings", []))
        image_hash = compute_image_hash(image)

        refresh_info = RefreshInfo(
            refresh_type="API Next",
            plugin_id=plugin_instance.plugin_id,
            refresh_time=now.isoformat(),
            image_hash=image_hash,
            playlist=playlist.name,
            plugin_instance=plugin_instance.name,
        )
        device_config.refresh_info = refresh_info
        device_config.write_config()

    payload = _build_status_payload(now=now, include_image=include_image, include_image_base64=include_image_base64)
    payload["action"] = {"type": "next", "playlist": playlist.name}
    return jsonify(payload)
