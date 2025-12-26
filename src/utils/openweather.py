from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)


WEATHER_URL = (
    "https://api.openweathermap.org/data/3.0/onecall?"
    "lat={lat}&lon={long}&units={units}&exclude=minutely&appid={api_key}"
)


@dataclass(frozen=True)
class WeatherSnapshot:
    now: datetime
    units: str
    current_temp: float
    feels_like: float
    description: str
    icon: str
    daily: list[dict[str, Any]]


_WEATHER_CACHE: dict[tuple[str, str, str, str], tuple[float, dict[str, Any]]] = {}


def fetch_weather_snapshot(
    api_key: str,
    units: str,
    lat: str,
    lon: str,
    now: datetime,
    *,
    cache_ttl_sec: int = 0,
) -> WeatherSnapshot:
    cache_key = (str(api_key), str(units), str(lat), str(lon))
    ttl = max(0, int(cache_ttl_sec or 0))
    if ttl > 0:
        cached = _WEATHER_CACHE.get(cache_key)
        if cached:
            cached_at, payload = cached
            if (now.timestamp() - cached_at) <= ttl:
                return WeatherSnapshot(
                    now=now,
                    units=str(payload.get("units") or units),
                    current_temp=float(payload.get("current_temp") or 0.0),
                    feels_like=float(payload.get("feels_like") or payload.get("current_temp") or 0.0),
                    description=str(payload.get("description") or "weather"),
                    icon=str(payload.get("icon") or "01d"),
                    daily=list(payload.get("daily") or []),
                )

    url = WEATHER_URL.format(lat=lat, long=lon, units=units, api_key=api_key)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.exception("OpenWeatherMap request failed: %s", exc)
        raise RuntimeError("OpenWeatherMap request failure, please check logs.") from exc

    current = data.get("current", {}) or {}
    weather_entry = (current.get("weather") or [{}])[0] or {}
    icon = str(weather_entry.get("icon") or "01d").replace("n", "d")
    description = str(weather_entry.get("description") or "").strip() or "weather"

    daily = data.get("daily") or []
    if not isinstance(daily, list):
        daily = []

    current_temp = float(current.get("temp") or 0.0)
    feels_like = float(current.get("feels_like") or current_temp)

    if ttl > 0:
        _WEATHER_CACHE[cache_key] = (
            now.timestamp(),
            {
                "units": units,
                "current_temp": current_temp,
                "feels_like": feels_like,
                "description": description,
                "icon": icon,
                "daily": daily,
            },
        )

    return WeatherSnapshot(
        now=now,
        units=units,
        current_temp=current_temp,
        feels_like=feels_like,
        description=description,
        icon=icon,
        daily=daily,
    )
