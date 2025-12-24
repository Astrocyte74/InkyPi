import json
import logging
import os
import tempfile
from typing import Any
import re

from flask import Blueprint, jsonify, render_template, request, current_app
from datetime import datetime
import pytz

from plugins.daily_cat_weather.daily_cat_weather import DailyCatWeather, THEME_CATALOG_PATH
from utils.family_events import DEFAULT_FAMILY_DATES_PATH, load_family_payload, pick_family_banner

logger = logging.getLogger(__name__)

themes_bp = Blueprint("themes", __name__)


def _load_theme_file(path: str) -> dict[str, Any]:
    if not path:
        return {"version": 1, "themes": []}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
            return payload if isinstance(payload, dict) else {"version": 1, "themes": []}
    except FileNotFoundError:
        return {"version": 1, "themes": []}
    except Exception:
        logger.exception("Failed to load theme file: %s", path)
        return {"version": 1, "themes": []}


def _parse_theme_list(
    payload: dict[str, Any],
    *,
    source: str,
    builtin_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    builtin_ids = builtin_ids or set()
    themes = payload.get("themes")
    if not isinstance(themes, list):
        themes = []

    parsed: dict[str, dict[str, Any]] = {}
    for theme in themes:
        if not isinstance(theme, dict):
            continue
        theme_id = DailyCatWeather._normalize_theme_id(theme.get("id"))
        if not theme_id:
            continue

        label = str(theme.get("label") or theme_id).strip() or theme_id
        character = theme.get("character") if isinstance(theme.get("character"), dict) else {}
        style = theme.get("style") if isinstance(theme.get("style"), dict) else {}
        parsed[theme_id] = {
            "id": theme_id,
            "label": label,
            "character": {
                "noun": str(character.get("noun") or "character").strip() or "character",
                "prompt": str(character.get("prompt") or "").strip(),
            },
            "style": {
                "prompt": str(style.get("prompt") or "").strip(),
                "avoid": str(style.get("avoid") or "").strip(),
            },
            "source": source,
            "is_override": theme_id in builtin_ids if source == "local" else False,
        }
    return parsed


def _local_theme_path() -> str:
    return DailyCatWeather._theme_local_path()


def _write_local_theme_file(payload: dict[str, Any]) -> None:
    local_path = _local_theme_path()
    directory = os.path.dirname(local_path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".themes.", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, local_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


def _clear_theme_cache() -> None:
    DailyCatWeather._theme_catalog_cache = None
    DailyCatWeather._theme_catalog_cache_key_value = None


def _validate_theme(theme: dict[str, Any]) -> dict[str, Any]:
    theme_id = DailyCatWeather._normalize_theme_id(theme.get("id"))
    if not theme_id:
        raise ValueError("Theme id is required (letters, numbers, underscore).")

    label = str(theme.get("label") or theme_id).strip() or theme_id
    character = theme.get("character") if isinstance(theme.get("character"), dict) else {}
    style = theme.get("style") if isinstance(theme.get("style"), dict) else {}

    return {
        "id": theme_id,
        "label": label,
        "character": {
            "noun": str(character.get("noun") or "character").strip() or "character",
            "prompt": str(character.get("prompt") or "").strip(),
        },
        "style": {
            "prompt": str(style.get("prompt") or "").strip(),
            "avoid": str(style.get("avoid") or "").strip(),
        },
    }


@themes_bp.route("/themes/daily_cat_weather")
def daily_cat_weather_editor():
    return render_template("theme_editor_daily_cat_weather.html")


@themes_bp.route("/api/themes/daily_cat_weather")
def api_daily_cat_weather_themes():
    builtin_payload = _load_theme_file(THEME_CATALOG_PATH)
    builtin = _parse_theme_list(builtin_payload, source="builtin")
    local_payload = _load_theme_file(_local_theme_path())
    local = _parse_theme_list(local_payload, source="local", builtin_ids=set(builtin.keys()))

    merged = dict(builtin)
    merged.update(local)

    return jsonify(
        {
            "default_theme_id": "storybook",
            "local_path": _local_theme_path(),
            "themes": list(merged.values()),
            "builtin_ids": sorted(builtin.keys()),
            "local_ids": sorted(local.keys()),
        }
    )


@themes_bp.route("/api/themes/daily_cat_weather", methods=["POST"])
def api_daily_cat_weather_save_theme():
    data = request.get_json(silent=True) or {}
    try:
        theme = data.get("theme")
        if not isinstance(theme, dict):
            return jsonify({"error": "Request must include a 'theme' object."}), 400
        cleaned = _validate_theme(theme)

        local_payload = _load_theme_file(_local_theme_path())
        local_themes = local_payload.get("themes")
        if not isinstance(local_themes, list):
            local_themes = []

        theme_id = cleaned["id"]
        replaced = False
        updated_list: list[dict[str, Any]] = []
        for item in local_themes:
            if not isinstance(item, dict):
                continue
            item_id = DailyCatWeather._normalize_theme_id(item.get("id"))
            if item_id == theme_id:
                updated_list.append(cleaned)
                replaced = True
            else:
                updated_list.append(item)

        if not replaced:
            updated_list.append(cleaned)

        local_payload = {"version": 1, "themes": updated_list}
        _write_local_theme_file(local_payload)
        _clear_theme_cache()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Failed to save theme")
        return jsonify({"error": "Failed to save theme."}), 500

    return jsonify({"success": True, "message": f"Saved theme '{theme_id}' to local overrides."})


@themes_bp.route("/api/themes/daily_cat_weather/<theme_id>", methods=["DELETE"])
def api_daily_cat_weather_delete_theme(theme_id: str):
    theme_id = DailyCatWeather._normalize_theme_id(theme_id)
    if not theme_id:
        return jsonify({"error": "Invalid theme id."}), 400

    try:
        local_payload = _load_theme_file(_local_theme_path())
        local_themes = local_payload.get("themes")
        if not isinstance(local_themes, list):
            local_themes = []

        updated_list: list[dict[str, Any]] = []
        removed = False
        for item in local_themes:
            if not isinstance(item, dict):
                continue
            item_id = DailyCatWeather._normalize_theme_id(item.get("id"))
            if item_id == theme_id:
                removed = True
                continue
            updated_list.append(item)

        if not removed:
            return jsonify({"error": f"No local theme '{theme_id}' to delete."}), 404

        local_payload = {"version": 1, "themes": updated_list}
        _write_local_theme_file(local_payload)
        _clear_theme_cache()
    except Exception:
        logger.exception("Failed to delete theme")
        return jsonify({"error": "Failed to delete theme."}), 500

    return jsonify({"success": True, "message": f"Deleted local theme '{theme_id}'."})


def _family_dates_path() -> str:
    device_config = current_app.config.get("DEVICE_CONFIG")
    if device_config:
        path = device_config.load_env_key("INKYPI_FAMILY_DATES_PATH")
        if path:
            path = path.strip()
            if len(path) >= 2 and path[0] == path[-1] and path[0] in {"'", '"', "`"}:
                path = path[1:-1].strip()
            return path
    return DEFAULT_FAMILY_DATES_PATH


def _write_json_atomic(path: str, payload: dict[str, Any], *, tmp_prefix: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=tmp_prefix, suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


def _validate_family_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config")
    config = config if isinstance(config, dict) else {}

    def _bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        return default

    def _int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    cleaned_config = {
        "banner_enabled": _bool(config.get("banner_enabled"), True),
        "lead_days": max(0, _int(config.get("lead_days"), 3)),
        "lead_days_max": max(0, _int(config.get("lead_days_max"), 14)),
        "show_birthday_age": _bool(config.get("show_birthday_age"), False),
        "show_anniversary_years": _bool(config.get("show_anniversary_years"), True),
    }

    items = payload.get("items")
    if not isinstance(items, list):
        items = []

    cleaned_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        date_str = str(item.get("date") or "").strip()
        text = str(item.get("text") or "").strip()
        if not re.fullmatch(r"\d{2}-\d{2}", date_str):
            continue
        if not date_str or not text:
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in {"birthday", "anniversary", "holiday"}:
            kind = "anniversary" if "anniversary" in text.lower() else "birthday"
        year = item.get("year")
        year_int = None
        if isinstance(year, int):
            year_int = year
        elif isinstance(year, str) and year.strip().isdigit():
            year_int = int(year.strip())
        cleaned_items.append({"date": date_str, "text": text, "year": year_int, "kind": kind})

    cleaned_items.sort(key=lambda r: (str(r.get("date") or ""), str(r.get("text") or "").lower()))

    return {
        "version": 1,
        "id": "family",
        "label": "Family Dates",
        "type": "family",
        "config": cleaned_config,
        "items": cleaned_items,
    }


@themes_bp.route("/themes/family_dates")
def family_dates_editor():
    return render_template("family_dates_editor.html")


@themes_bp.route("/api/family_dates")
def api_family_dates():
    path = _family_dates_path()
    payload = load_family_payload(path) or {}
    config = payload.get("config") if isinstance(payload, dict) else {}
    config = config if isinstance(config, dict) else {}
    items = payload.get("items") if isinstance(payload, dict) else []
    items = items if isinstance(items, list) else []

    device_config = current_app.config.get("DEVICE_CONFIG")
    tz_str = device_config.get_config("timezone", default="UTC") if device_config else "UTC"
    tz = pytz.timezone(tz_str)
    now = datetime.now(tz)

    config_defaults = {
        "banner_enabled": True,
        "lead_days": 3,
        "lead_days_max": 14,
        "show_birthday_age": False,
        "show_anniversary_years": True,
    }
    merged_config = dict(config_defaults)
    merged_config.update({k: v for k, v in config.items() if k in config_defaults})

    # Compute a preview using JSON config only (env overrides are handled in runtime rendering).
    lead_days = int(merged_config.get("lead_days") or 3)
    max_lead = int(merged_config.get("lead_days_max") or 14)
    show_bday = bool(merged_config.get("show_birthday_age"))
    show_ann = bool(merged_config.get("show_anniversary_years"))

    from utils.family_events import load_family_events

    events = load_family_events(path)
    preview = pick_family_banner(
        now=now,
        events=events,
        lead_days=lead_days,
        max_lead_days=max_lead,
        show_birthday_age=show_bday,
        show_anniversary_years=show_ann,
    )

    return jsonify(
        {
            "path": path,
            "config": merged_config,
            "items": items,
            "preview": preview,
            "timezone": tz_str,
            "now": now.isoformat(),
        }
    )


@themes_bp.route("/api/family_dates", methods=["POST"])
def api_family_dates_save():
    path = _family_dates_path()
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON object."}), 400
    try:
        cleaned = _validate_family_payload(data)
        _write_json_atomic(path, cleaned, tmp_prefix=".family_dates.")
        # Clear cache.
        from utils import family_events as fe

        fe._PAYLOAD_CACHE.pop(path, None)  # pylint: disable=protected-access
        fe._EVENT_CACHE.pop(path, None)  # pylint: disable=protected-access
    except Exception as exc:
        logger.exception("Failed to save family dates")
        return jsonify({"error": f"Failed to save: {exc}"}), 500
    return jsonify({"success": True, "message": f"Saved family dates to {path}."})


@themes_bp.route("/api/family_dates/preview")
def api_family_dates_preview():
    path = _family_dates_path()
    payload = load_family_payload(path) or {}
    config = payload.get("config") if isinstance(payload, dict) else {}
    config = config if isinstance(config, dict) else {}
    config_defaults = {
        "lead_days": 3,
        "lead_days_max": 14,
        "show_birthday_age": False,
        "show_anniversary_years": True,
    }
    merged_config = dict(config_defaults)
    merged_config.update({k: v for k, v in config.items() if k in config_defaults})

    device_config = current_app.config.get("DEVICE_CONFIG")
    tz_str = device_config.get_config("timezone", default="UTC") if device_config else "UTC"
    tz = pytz.timezone(tz_str)

    date_str = (request.args.get("date") or "").strip()
    if date_str:
        try:
            now = tz.localize(datetime.fromisoformat(date_str))
        except Exception:
            return jsonify({"error": "Invalid date; use YYYY-MM-DD."}), 400
    else:
        now = datetime.now(tz)

    lead_days = int(merged_config.get("lead_days") or 3)
    max_lead = int(merged_config.get("lead_days_max") or 14)
    show_bday = bool(merged_config.get("show_birthday_age"))
    show_ann = bool(merged_config.get("show_anniversary_years"))

    from utils.family_events import load_family_events

    events = load_family_events(path)
    preview = pick_family_banner(
        now=now,
        events=events,
        lead_days=lead_days,
        max_lead_days=max_lead,
        show_birthday_age=show_bday,
        show_anniversary_years=show_ann,
    )

    return jsonify(
        {
            "preview": preview,
            "now": now.isoformat(),
            "timezone": tz_str,
        }
    )
