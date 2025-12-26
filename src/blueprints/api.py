from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timedelta
from contextlib import nullcontext
from datetime import time as dtime
import re

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


def _parse_hhmm(value: str) -> dtime:
    value = (value or "").strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
    if not match:
        return dtime(4, 0)
    hour = max(0, min(23, int(match.group(1))))
    minute = max(0, min(59, int(match.group(2))))
    return dtime(hour, minute)


def _day_key(now: datetime, refresh_time: dtime) -> str:
    if now.timetz().replace(tzinfo=None) < refresh_time:
        day = (now - timedelta(days=1)).date()
    else:
        day = now.date()
    return day.isoformat()


def _sanitize_cache_id(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "default"


def _clear_daily_cat_cache(device_config, plugin_instance, now: datetime):
    settings = plugin_instance.settings or {}
    cache_id = _sanitize_cache_id(settings.get("cacheId") or "default")
    daily_refresh_time = _parse_hhmm(settings.get("dailyRefreshTime") or "04:00")
    day = _day_key(now, daily_refresh_time)

    cache_dir = os.path.join(device_config.BASE_DIR, "..", "mock_display_output", "daily_cat_weather")
    os.makedirs(cache_dir, exist_ok=True)

    targets = [
        os.path.join(cache_dir, f"bg_{cache_id}_{day}.png"),
        os.path.join(cache_dir, f"bg_{cache_id}_{day}.json"),
        os.path.join(cache_dir, f"latest_bg_{cache_id}.png"),
    ]

    removed = []
    for path in targets:
        if not os.path.exists(path):
            continue
        try:
            os.remove(path)
            removed.append(os.path.basename(path))
        except Exception:
            logger.exception("Failed to remove Daily Cat cache file: %s", path)

    return cache_id, day, removed


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

    image_url = None
    try:
        if current_path and os.path.exists(current_path):
            # Stable URL for iOS Shortcuts (smaller/faster than base64 for large payloads).
            base = request.host_url.rstrip("/")
            cache_bust = ""
            if last_refresh_iso:
                cache_bust = f"?t={last_refresh_iso}"
            image_url = f"{base}/static/images/current_image.png{cache_bust}"
    except Exception:
        image_url = None

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
            "url": image_url,
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
    fmt = str(request.args.get("format", "base64")).strip().lower()
    include_image_base64 = fmt in {"base64", "b64", "json"}

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
    force_raw = str(request.args.get("force", "")).strip().lower()
    force_mode = "smart"
    if force_raw in {"1", "true", "yes", "all"}:
        force_mode = "all"
    elif force_raw in {"0", "false", "no", "none"}:
        force_mode = "none"

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

        # Default behavior ("smart"):
        # - Always regenerate "light" slides like flags/quotes (so you don't see repeats)
        # - Do NOT regenerate expensive AI slides like the daily illustration unless explicitly forced.
        force_refresh = False
        if force_mode == "all":
            force_refresh = True
        elif force_mode == "none":
            force_refresh = False
        else:
            force_refresh = plugin_instance.plugin_id in {"world_flags", "daily_theme_card"}

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
    payload["action"] = {
        "type": "next",
        "playlist": playlist.name,
        "force_mode": force_mode,
        "forced_refresh": bool(force_refresh),
    }
    return jsonify(payload)


@api_bp.route("/cat/reroll", methods=["GET", "POST"])
def cat_reroll():
    """Force a new Daily Cat background and make it the "today" image."""
    if not _require_token():
        return jsonify({"error": "Unauthorized"}), 401

    include_image = str(request.args.get("image", "1")).strip().lower() not in {"0", "false", "no"}
    include_image_base64 = str(request.args.get("format", "base64")).strip().lower() in {"base64", "b64", "json"}
    requested_instance = (request.args.get("instance") or "").strip()

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

    plugin_instance = None
    if requested_instance:
        plugin_instance = playlist.find_plugin("daily_cat_weather", requested_instance)

    if not plugin_instance:
        try:
            refresh_info = device_config.get_refresh_info()
            if refresh_info and getattr(refresh_info, "plugin_id", None) == "daily_cat_weather":
                current_name = getattr(refresh_info, "plugin_instance", None)
                if current_name:
                    plugin_instance = playlist.find_plugin("daily_cat_weather", current_name)
        except Exception:
            plugin_instance = None

    if not plugin_instance:
        plugin_instance = next((p for p in playlist.plugins if p.plugin_id == "daily_cat_weather"), None)

    if not plugin_instance:
        return jsonify({"error": "Daily Cat Weather isn't enabled in the active playlist."}), 400

    lock = getattr(refresh_task, "lock", None)
    ctx = lock if lock is not None else nullcontext()

    from plugins.plugin_registry import get_plugin_instance
    from refresh_task import PlaylistRefresh
    from utils.image_utils import compute_image_hash
    from model import RefreshInfo

    with ctx:
        plugin_instance.settings = plugin_instance.settings or {}
        try:
            current_nonce = int(plugin_instance.settings.get("rerollNonce") or 0)
        except (TypeError, ValueError):
            current_nonce = 0
        plugin_instance.settings["rerollNonce"] = current_nonce + 1

        cache_id, day, removed = _clear_daily_cat_cache(device_config, plugin_instance, now)
        try:
            device_config.write_config()
        except Exception:
            logger.exception("Failed to persist Daily Cat settings.")

        plugin_config = device_config.get_plugin(plugin_instance.plugin_id)
        if not plugin_config:
            return jsonify({"error": "Daily Cat Weather plugin config not found."}), 404

        plugin = get_plugin_instance(plugin_config)
        image = PlaylistRefresh(playlist, plugin_instance, force=True).execute(plugin, device_config, now)

        display_manager.display_image(image, image_settings=plugin_config.get("image_settings", []))
        image_hash = compute_image_hash(image)

        refresh_info = RefreshInfo(
            refresh_type="API Cat Reroll",
            plugin_id=plugin_instance.plugin_id,
            refresh_time=now.isoformat(),
            image_hash=image_hash,
            playlist=playlist.name,
            plugin_instance=plugin_instance.name,
        )
        device_config.refresh_info = refresh_info
        device_config.write_config()

    payload = _build_status_payload(now=now, include_image=include_image, include_image_base64=include_image_base64)
    payload["action"] = {
        "type": "cat_reroll",
        "playlist": playlist.name,
        "plugin_instance": plugin_instance.name,
        "cache_id": cache_id,
        "day": day,
        "reroll_nonce": plugin_instance.settings.get("rerollNonce"),
        "removed_cache_files": removed,
    }
    return jsonify(payload)
