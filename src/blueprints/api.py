from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timedelta
from contextlib import nullcontext
from datetime import time as dtime
import re
import pytz
import threading

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


def _set_banner_override_from_text(device_config, text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Banner text is empty.")
    parts = [p.strip() for p in raw.splitlines() if p.strip()]
    headline = parts[0] if parts else raw
    detail = ""
    if len(parts) >= 2:
        detail = " ".join(parts[1:]).strip()

    headline = headline[:120].rstrip()
    detail = detail[:180].rstrip()

    tz_str = device_config.get_config("timezone", default="UTC")
    try:
        tz = pytz.timezone(tz_str)
    except Exception:
        tz = pytz.UTC
    day = datetime.now(tz).date().isoformat()

    cfg = device_config.get_config()
    cfg["banner_override"] = {
        "day": day,
        "headline": headline,
        "detail": detail,
    }
    device_config.update_config(cfg)
    return cfg["banner_override"]


def _clear_banner_override(device_config) -> bool:
    cfg = device_config.get_config()
    if "banner_override" not in cfg:
        return False
    cfg.pop("banner_override", None)
    device_config.update_config(cfg)
    return True


def _mark_banner_consumers_stale(device_config, now: datetime) -> None:
    try:
        playlist_manager = device_config.get_playlist_manager()
        playlist = playlist_manager.determine_active_playlist(now)
    except Exception:
        playlist = None

    if not playlist or not getattr(playlist, "plugins", None):
        return

    changed = False
    for plugin_instance in playlist.plugins:
        if plugin_instance.plugin_id in {"daily_cat_weather", "daily_theme_card"}:
            plugin_instance.latest_refresh_time = None
            changed = True

    if changed:
        try:
            device_config.write_config()
        except Exception:
            logger.exception("Failed to persist playlist after banner update.")


def _refresh_current_banner_slide(*, device_config, display_manager, refresh_task, now: datetime) -> bool:
    try:
        refresh_info = device_config.get_refresh_info()
    except Exception:
        refresh_info = None

    plugin_id = getattr(refresh_info, "plugin_id", None) if refresh_info else None
    plugin_instance_name = getattr(refresh_info, "plugin_instance", None) if refresh_info else None
    if plugin_id not in {"daily_cat_weather", "daily_theme_card"} or not plugin_instance_name:
        return False

    try:
        playlist_manager = device_config.get_playlist_manager()
        playlist = playlist_manager.determine_active_playlist(now)
    except Exception:
        playlist = None
    if not playlist:
        return False

    plugin_instance = playlist.find_plugin(plugin_id, plugin_instance_name)
    if not plugin_instance:
        return False

    if not getattr(refresh_task, "running", False):
        return False

    from plugins.plugin_registry import get_plugin_instance
    from refresh_task import PlaylistRefresh
    from utils.image_utils import compute_image_hash
    from model import RefreshInfo

    lock = getattr(refresh_task, "lock", None)
    ctx = lock if lock is not None else nullcontext()

    with ctx:
        plugin_config = device_config.get_plugin(plugin_instance.plugin_id)
        if not plugin_config:
            return False
        plugin = get_plugin_instance(plugin_config)
        image = PlaylistRefresh(playlist, plugin_instance, force=True).execute(plugin, device_config, now)
        display_manager.display_image(image, image_settings=plugin_config.get("image_settings", []))
        image_hash = compute_image_hash(image)
        device_config.refresh_info = RefreshInfo(
            refresh_type="API Banner",
            plugin_id=plugin_instance.plugin_id,
            refresh_time=now.isoformat(),
            image_hash=image_hash,
            playlist=playlist.name,
            plugin_instance=plugin_instance.name,
        )
        device_config.write_config()
    return True


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


@api_bp.route("/banner", methods=["GET", "POST"])
def banner():
    """
    Set or clear the top banner override (Shortcuts-friendly).

    - POST /api/banner?text=Hello
    - POST /api/banner?clear=1
    """
    if not _require_token():
        return jsonify({"error": "Unauthorized"}), 401

    include_image = str(request.args.get("image", "1")).strip().lower() not in {"0", "false", "no"}
    include_image_base64 = str(request.args.get("format", "base64")).strip().lower() in {"base64", "b64", "json"}

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

    data = request.get_json(silent=True) if request.method == "POST" else None
    clear_raw = request.args.get("clear")
    clear = False
    if clear_raw is not None:
        clear = str(clear_raw).strip().lower() in {"1", "true", "yes", "on"}
    elif isinstance(data, dict) and "clear" in data:
        clear = bool(data.get("clear"))

    action = {"type": "banner"}
    if clear:
        removed = _clear_banner_override(device_config)
        action.update({"op": "clear", "removed": bool(removed)})
    else:
        text = request.args.get("text")
        if text is None and isinstance(data, dict):
            text = data.get("text")
        text = (text or "").strip()
        if not text:
            return jsonify({"error": "Missing 'text' (or use clear=1)."}), 400
        override = _set_banner_override_from_text(device_config, text)
        action.update({"op": "set", "headline": override.get("headline"), "detail": override.get("detail")})

    _mark_banner_consumers_stale(device_config, now)

    refresh_raw = request.args.get("refresh")
    refresh_now = False
    if refresh_raw is not None:
        refresh_now = str(refresh_raw).strip().lower() in {"1", "true", "yes", "on"}
    elif isinstance(data, dict) and "refresh" in data:
        refresh_now = bool(data.get("refresh"))

    # IMPORTANT: Refreshing the current slide can take a long time on real e-ink hardware.
    # Keep the API responsive by default; allow optional async refresh.
    action["refresh_requested"] = bool(refresh_now)
    action["refreshed"] = False
    action["refresh_queued"] = False
    if refresh_now:
        def _worker():
            try:
                _refresh_current_banner_slide(
                    device_config=device_config,
                    display_manager=display_manager,
                    refresh_task=refresh_task,
                    now=now,
                )
            except Exception:
                logger.exception("API banner refresh worker failed.")

        threading.Thread(target=_worker, name="ApiBannerRefresh", daemon=True).start()
        action["refresh_queued"] = True

    payload = _build_status_payload(now=now, include_image=include_image, include_image_base64=include_image_base64)
    payload["action"] = action
    return jsonify(payload)


@api_bp.route("/ai", methods=["GET", "POST"])
def ai_generate():
    """
    Generate a new image from an idea and display it using the standard 2-panel layout.

    Modes:
    - Temporary (default): generate and display a one-off slide (does not change daily_cat_weather settings).
    - Today: set the idea as today's Daily Cat custom prompt and reroll (persists as the illustration-of-the-day).
    """
    if not _require_token():
        return jsonify({"error": "Unauthorized"}), 401

    include_image = str(request.args.get("image", "1")).strip().lower() not in {"0", "false", "no"}
    include_image_base64 = str(request.args.get("format", "base64")).strip().lower() in {"base64", "b64", "json"}

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

    data = request.get_json(silent=True) if request.method == "POST" else None
    idea = request.args.get("idea")
    if idea is None and isinstance(data, dict):
        idea = data.get("idea")
    idea = (idea or "").strip()
    if not idea:
        return jsonify({"error": "Missing 'idea'."}), 400

    enhance_raw = request.args.get("enhance")
    enhance = False
    if enhance_raw is not None:
        enhance = str(enhance_raw).strip().lower() in {"1", "true", "yes", "on"}
    elif isinstance(data, dict) and "enhance" in data:
        enhance = bool(data.get("enhance"))

    today_raw = request.args.get("today")
    today = False
    if today_raw is not None:
        today = str(today_raw).strip().lower() in {"1", "true", "yes", "on"}
    elif isinstance(data, dict) and "today" in data:
        today = bool(data.get("today"))

    playlist_manager = device_config.get_playlist_manager()
    playlist = playlist_manager.determine_active_playlist(now)
    if not playlist or not playlist.plugins:
        return jsonify({"error": "No active playlist with plugins."}), 400

    # Find the Daily Cat instance, to source weather + (optionally) persist "today" prompt.
    cat_instance = next((p for p in playlist.plugins if p.plugin_id == "daily_cat_weather"), None)
    cat_settings = (cat_instance.settings or {}) if cat_instance else {}

    if today:
        if not cat_instance:
            return jsonify({"error": "Daily Cat Weather isn't enabled in the active playlist."}), 400

        # Optional prompt enhancement via OpenRouter/OpenAI (same helper used by AIImage).
        enhanced = ""
        if enhance:
            try:
                from plugins.ai_image.ai_image import AIImage
                from openai import OpenAI
                openai_key = (device_config.load_env_key("OPEN_AI_SECRET") or "").strip()
                openrouter_key = (device_config.load_env_key("OPEN_ROUTER_SECRET") or "").strip()
                if openrouter_key:
                    prompt_client = {
                        "type": "openrouter",
                        "api_key": openrouter_key,
                        "model": AIImage._resolve_openrouter_model(device_config.load_env_key("OPEN_ROUTER_MODEL")),
                        "referer": device_config.load_env_key("OPEN_ROUTER_REFERRER") or "https://github.com/fatihak/InkyPi",
                        "title": device_config.load_env_key("OPEN_ROUTER_TITLE") or "InkyPi",
                    }
                    enhanced = (AIImage.enhance_prompt(prompt_client, idea) or "").strip()
                elif openai_key:
                    prompt_client = {"type": "openai", "client": OpenAI(api_key=openai_key)}
                    enhanced = (AIImage.enhance_prompt(prompt_client, idea) or "").strip()
            except Exception:
                logger.exception("AI prompt enhancement failed; using raw idea.")
                enhanced = ""

        cat_instance.settings = cat_instance.settings or {}
        try:
            current_nonce = int(cat_instance.settings.get("rerollNonce") or 0)
        except (TypeError, ValueError):
            current_nonce = 0
        cat_instance.settings["rerollNonce"] = current_nonce + 1

        daily_refresh_time = _parse_hhmm(cat_instance.settings.get("dailyRefreshTime") or "04:00")
        day = _day_key(now, daily_refresh_time)
        cat_instance.settings["customPrompt"] = idea
        if enhanced:
            cat_instance.settings["customPromptEnhanced"] = enhanced
        else:
            cat_instance.settings.pop("customPromptEnhanced", None)
        cat_instance.settings["customPromptDayKey"] = day

        cache_id, day_key, removed = _clear_daily_cat_cache(device_config, cat_instance, now)
        try:
            device_config.write_config()
        except Exception:
            logger.exception("Failed to persist Daily Cat settings.")

        # Force refresh of the cat slide now.
        from plugins.plugin_registry import get_plugin_instance
        from refresh_task import PlaylistRefresh
        from utils.image_utils import compute_image_hash
        from model import RefreshInfo

        lock = getattr(refresh_task, "lock", None)
        ctx = lock if lock is not None else nullcontext()
        with ctx:
            plugin_config = device_config.get_plugin(cat_instance.plugin_id)
            if not plugin_config:
                return jsonify({"error": "Daily Cat Weather plugin config not found."}), 404
            plugin = get_plugin_instance(plugin_config)
            image = PlaylistRefresh(playlist, cat_instance, force=True).execute(plugin, device_config, now)
            display_manager.display_image(image, image_settings=plugin_config.get("image_settings", []))
            image_hash = compute_image_hash(image)
            device_config.refresh_info = RefreshInfo(
                refresh_type="API AI Today",
                plugin_id=cat_instance.plugin_id,
                refresh_time=now.isoformat(),
                image_hash=image_hash,
                playlist=playlist.name,
                plugin_instance=cat_instance.name,
            )
            device_config.write_config()

        payload = _build_status_payload(now=now, include_image=include_image, include_image_base64=include_image_base64)
        payload["action"] = {
            "type": "ai",
            "mode": "today",
            "idea": idea,
            "enhanced": enhanced or None,
            "cache_id": cache_id,
            "day": day_key,
            "reroll_nonce": cat_instance.settings.get("rerollNonce"),
            "removed_cache_files": removed,
        }
        return jsonify(payload)

    # --- Temporary mode (one-off slide) -----------------------------------
    from plugins.plugin_registry import get_plugin_instance
    from utils.openweather import fetch_weather_snapshot
    from utils.weather_sidebar import render_weather_sidebar_panel
    from utils.family_events import banner_from_env
    from utils.family_banner import draw_family_banner
    from utils.image_utils import compute_image_hash
    from model import RefreshInfo
    from PIL import Image

    model = request.args.get("model")
    if model is None and isinstance(data, dict):
        model = data.get("model")
    model = (model or "").strip()
    if not model:
        model = (device_config.load_env_key("TELEGRAM_AI_DEFAULT_MODEL") or "").split(",")[0].strip() or "gemini-2.5-flash-image"

    palette = request.args.get("palette")
    if palette is None and isinstance(data, dict):
        palette = data.get("palette")
    palette = (palette or "spectra6").strip().lower()
    if palette not in {"spectra6", "bw"}:
        palette = "spectra6"

    style_hint = request.args.get("style")
    if style_hint is None and isinstance(data, dict):
        style_hint = data.get("style")
    style_hint = (style_hint or "").strip().lower()

    settings = {
        "textPrompt": idea,
        "imageModel": model,
        "quality": (request.args.get("quality") or "2k").strip().lower(),
        "palette": palette,
        "randomizePrompt": "false",
        "creativeEnhance": "true" if enhance else "false",
        "styleHint": style_hint,
        "vanGoghStyle": "false",
    }

    plugin_config = device_config.get_plugin("ai_image")
    if not plugin_config:
        return jsonify({"error": "AI Image plugin is not available."}), 500
    ai_plugin = get_plugin_instance(plugin_config)
    try:
        raw = ai_plugin.generate_image(settings, device_config).convert("RGB")
    except RuntimeError as exc:
        # Keep iOS Shortcuts-friendly JSON (avoid HTML 500 pages).
        msg = str(exc) or "AI generation failed."
        return jsonify({"error": msg}), 400
    except Exception as exc:
        logger.exception("API AI generation failed: %s", exc)
        return jsonify({"error": "AI generation failed, please check logs."}), 500

    # Determine sizes (match the Daily Cat layout ratio).
    try:
        from plugins.daily_cat_weather.daily_cat_weather import SIDEBAR_WIDTH_RATIO, DailyCatWeather
        sidebar_ratio = float(SIDEBAR_WIDTH_RATIO)
        cover_crop = DailyCatWeather._cover_crop  # pylint: disable=protected-access
        font_fn = DailyCatWeather._font  # pylint: disable=protected-access
        icon_renderer = DailyCatWeather._simple_weather_icon  # pylint: disable=protected-access
    except Exception:
        sidebar_ratio = 0.30
        cover_crop = None
        font_fn = None
        icon_renderer = None

    width, height = device_config.get_resolution()
    if device_config.get_config("orientation") == "vertical":
        width, height = height, width

    sidebar_width = int(width * sidebar_ratio)
    sidebar_width = max(120, min(width - 100, sidebar_width))
    image_width = width - sidebar_width

    if cover_crop:
        left_img = cover_crop(raw, (image_width, height))
    else:
        left_img = raw.resize((image_width, height))

    # Weather sidebar (best-effort using Daily Cat settings for location/units).
    weather = None
    try:
        owm_key = (device_config.load_env_key("OPEN_WEATHER_MAP_SECRET") or "").strip()
        lat = (cat_settings.get("latitude") or "").strip()
        lon = (cat_settings.get("longitude") or "").strip()
        units = (cat_settings.get("units") or "metric").strip().lower()
        forecast_days = int(cat_settings.get("forecastDays") or 3)
        forecast_days = max(1, min(5, forecast_days))
        ttl_min = int(cat_settings.get("weatherCacheMinutes") or 30)
        ttl_min = max(0, min(1440, ttl_min))
        if owm_key and lat and lon:
            weather = fetch_weather_snapshot(
                api_key=owm_key,
                units=units,
                lat=lat,
                lon=lon,
                now=now,
                cache_ttl_sec=ttl_min * 60,
            )
    except Exception:
        logger.exception("Failed to fetch weather for /api/ai; continuing without weather.")
        weather = None

    tz_str = device_config.get_config("timezone", default="UTC")
    try:
        tz = pytz.timezone(tz_str)
    except Exception:
        tz = pytz.UTC

    location_label = (cat_settings.get("locationLabel") or "").strip()
    if font_fn and icon_renderer:
        sidebar = render_weather_sidebar_panel(
            weather,
            tz,
            int(cat_settings.get("forecastDays") or 3),
            (sidebar_width, height),
            location_label,
            font=font_fn,
            icon_renderer=icon_renderer,
        )
    else:
        sidebar = Image.new("RGB", (sidebar_width, height), (255, 255, 255))

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(left_img, (0, 0))

    # Optional family banner.
    try:
        banner = banner_from_env(device_config, now=now)
        if banner:
            draw_family_banner(
                canvas,
                headline=banner.get("headline") or "",
                detail=banner.get("detail") or "",
                font_fn=font_fn,
                region=(0, 0, image_width, height),
            )
    except Exception:
        logger.exception("Failed to render family banner for /api/ai.")

    canvas.paste(sidebar, (image_width, 0))

    # Display and record refresh info.
    display_manager.display_image(canvas)
    image_hash = compute_image_hash(canvas)
    device_config.refresh_info = RefreshInfo(
        refresh_type="API AI Temp",
        plugin_id="api_ai",
        refresh_time=now.isoformat(),
        image_hash=image_hash,
        playlist=getattr(playlist, "name", None),
        plugin_instance=None,
    )
    device_config.write_config()

    payload = _build_status_payload(now=now, include_image=include_image, include_image_base64=include_image_base64)
    payload["action"] = {
        "type": "ai",
        "mode": "temporary",
        "idea": idea,
        "model": model,
        "enhance": bool(enhance),
        "style": style_hint or None,
        "palette": palette,
    }
    return jsonify(payload)
