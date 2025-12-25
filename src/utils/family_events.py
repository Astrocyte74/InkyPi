from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)


DEFAULT_FAMILY_DATES_PATH = "/usr/local/inkypi/family_dates.json"

_PAYLOAD_CACHE: dict[str, tuple[float, dict]] = {}
_EVENT_CACHE: dict[str, tuple[float, list["FamilyEvent"]]] = {}


@dataclass(frozen=True)
class FamilyEvent:
    mmdd: str
    text: str
    year: int | None = None
    kind: str = ""


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _infer_kind(text: str, kind: str) -> str:
    kind = (kind or "").strip().lower()
    if kind in {"birthday", "anniversary", "holiday"}:
        return kind
    return "anniversary" if "anniversary" in (text or "").lower() else "birthday"


def load_family_events(path: str) -> list[FamilyEvent]:
    path = (path or "").strip() or DEFAULT_FAMILY_DATES_PATH
    if len(path) >= 2 and path[0] == path[-1] and path[0] in {"'", '"', "`"}:
        path = path[1:-1].strip()
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return []
    except Exception:
        logger.exception("Failed to stat family dates file: %s", path)
        return []

    cached = _EVENT_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    payload = load_family_payload(path)
    if not payload:
        return []

    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []

    events: list[FamilyEvent] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mmdd = str(item.get("date") or "").strip()
        if not re.fullmatch(r"\d{2}-\d{2}", mmdd):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        year_val = item.get("year")
        year = None
        if isinstance(year_val, int):
            year = year_val
        elif isinstance(year_val, str) and year_val.strip().isdigit():
            year = int(year_val.strip())
        kind = _infer_kind(text, str(item.get("kind") or ""))
        events.append(FamilyEvent(mmdd=mmdd, text=text, year=year, kind=kind))

    _EVENT_CACHE[path] = (mtime, events)
    return events


def load_family_payload(path: str) -> dict:
    path = (path or "").strip() or DEFAULT_FAMILY_DATES_PATH
    if len(path) >= 2 and path[0] == path[-1] and path[0] in {"'", '"', "`"}:
        path = path[1:-1].strip()
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("Failed to stat family dates file: %s", path)
        return {}

    cached = _PAYLOAD_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
            payload = payload if isinstance(payload, dict) else {}
    except Exception:
        logger.exception("Failed to read family dates file: %s", path)
        return {}

    _PAYLOAD_CACHE[path] = (mtime, payload)
    return payload


def load_family_config(path: str) -> dict:
    payload = load_family_payload(path)
    config = payload.get("config") if isinstance(payload, dict) else None
    return config if isinstance(config, dict) else {}


def _next_occurrence(mmdd: str, today: date) -> date | None:
    try:
        month, day = [int(p) for p in mmdd.split("-", 1)]
        candidate = date(today.year, month, day)
        if candidate < today:
            candidate = date(today.year + 1, month, day)
        return candidate
    except Exception:
        return None


def _format_age_or_years(event: FamilyEvent, when: date, *, show_birthday_age: bool, show_anniversary_years: bool) -> str:
    if event.kind == "holiday":
        return ""
    if not event.year:
        return ""

    years = when.year - event.year
    if years <= 0:
        return ""

    if event.kind == "anniversary":
        return str(years) if show_anniversary_years else ""
    return str(years) if show_birthday_age else ""


def _possessive(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return f"{text}'" if text[-1].lower() == "s" else f"{text}'s"


def _event_subject(event: FamilyEvent) -> str:
    text = (event.text or "").strip()
    if event.kind == "anniversary" and text.lower().endswith("anniversary"):
        base = text[: -len("anniversary")].strip(" -–—")
        base = base.rstrip()
        return base or text
    return text


def pick_family_banner(
    *,
    now: datetime,
    events: list[FamilyEvent],
    lead_days: int = 3,
    max_lead_days: int = 14,
    show_birthday_age: bool = False,
    show_anniversary_years: bool = True,
) -> dict[str, str] | None:
    max_lead_days = max(0, min(3660, int(max_lead_days)))
    lead_days = max(0, min(max_lead_days, int(lead_days)))
    today = now.date()

    candidates: list[tuple[int, date, FamilyEvent]] = []
    for event in events:
        when = _next_occurrence(event.mmdd, today)
        if not when:
            continue
        delta = (when - today).days
        if 0 <= delta <= lead_days:
            candidates.append((delta, when, event))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[2].text.lower()))
    delta, when, event = candidates[0]

    subject = _event_subject(event)
    count = _format_age_or_years(
        event,
        when,
        show_birthday_age=show_birthday_age,
        show_anniversary_years=show_anniversary_years,
    )
    count_suffix = f" ({count})" if count else ""

    if event.kind == "holiday":
        subject = subject or "Holiday"
        token = subject.lower()
        emoji = "🎉"
        if "christmas" in token or "xmas" in token:
            emoji = "🎄"
        elif "new year" in token:
            emoji = "🎉"
        elif "thanksgiving" in token:
            emoji = "🦃"
        elif "halloween" in token:
            emoji = "🎃"
        elif "easter" in token:
            emoji = "🐣"

        if delta == 0:
            if "christmas" in token or "xmas" in token:
                headline = f"{emoji} Merry Christmas!"
            else:
                headline = f"{emoji} Happy {subject}!"
        elif delta == 1:
            headline = f"{emoji} {subject} is tomorrow"
        else:
            headline = f"{emoji} {delta} days until {subject}"
    elif event.kind == "anniversary":
        emoji = "💍"
        label = f"{_possessive(subject)} Anniversary{count_suffix}".strip()
        if delta == 0:
            headline = f"{emoji} Happy Anniversary {subject}{count_suffix}!"
        elif delta == 1:
            headline = f"{emoji} {label} tomorrow"
        else:
            headline = f"{emoji} {label} in {delta} days"
    else:
        emoji = "🎂"
        label = f"{_possessive(subject)} Birthday{count_suffix}".strip()
        if delta == 0:
            headline = f"{emoji} Happy Birthday {subject}{count_suffix}!"
        elif delta == 1:
            headline = f"{emoji} {label} tomorrow"
        else:
            headline = f"{emoji} {label} in {delta} days"

    return {"headline": headline, "detail": ""}  # single-line banner text


def banner_from_env(device_config, *, now: datetime) -> dict[str, str] | None:
    # Telegram/WebUI banner override: when set, it takes precedence for the current local day.
    try:
        override = device_config.get_config("banner_override", default=None)
    except Exception:
        override = None
    if isinstance(override, dict):
        day = str(override.get("day") or "").strip()
        if day and day == now.date().isoformat():
            headline = str(override.get("headline") or "").strip()
            detail = str(override.get("detail") or "").strip()
            if headline:
                return {"headline": headline, "detail": detail}

    path = device_config.load_env_key("INKYPI_FAMILY_DATES_PATH") or DEFAULT_FAMILY_DATES_PATH
    config = load_family_config(path)

    enabled_default = config.get("banner_enabled")
    enabled_default = bool(enabled_default) if isinstance(enabled_default, bool) else True

    enabled_raw = device_config.load_env_key("INKYPI_FAMILY_BANNER_ENABLED")
    enabled = _parse_bool(enabled_raw, enabled_default)
    if not enabled:
        return None

    events = load_family_events(path)
    if not events:
        return None

    lead_default = config.get("lead_days") if isinstance(config.get("lead_days"), int) else 3
    lead_raw = device_config.load_env_key("INKYPI_FAMILY_BANNER_LEAD_DAYS")
    lead_days = _parse_int(lead_raw, lead_default)

    max_default = config.get("lead_days_max") if isinstance(config.get("lead_days_max"), int) else 14
    max_raw = device_config.load_env_key("INKYPI_FAMILY_BANNER_LEAD_DAYS_MAX")
    max_lead_days = _parse_int(max_raw, max_default)

    show_birth_default = config.get("show_birthday_age")
    show_birth_default = bool(show_birth_default) if isinstance(show_birth_default, bool) else False
    show_birth_raw = device_config.load_env_key("INKYPI_FAMILY_SHOW_BIRTHDAY_AGE")
    show_birthday_age = _parse_bool(show_birth_raw, show_birth_default)

    show_ann_default = config.get("show_anniversary_years")
    show_ann_default = bool(show_ann_default) if isinstance(show_ann_default, bool) else True
    show_ann_raw = device_config.load_env_key("INKYPI_FAMILY_SHOW_ANNIVERSARY_YEARS")
    show_anniversary_years = _parse_bool(show_ann_raw, show_ann_default)

    return pick_family_banner(
        now=now,
        events=events,
        lead_days=lead_days,
        max_lead_days=max_lead_days,
        show_birthday_age=show_birthday_age,
        show_anniversary_years=show_anniversary_years,
    )
