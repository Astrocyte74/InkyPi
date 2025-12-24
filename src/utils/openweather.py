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


def fetch_weather_snapshot(api_key: str, units: str, lat: str, lon: str, now: datetime) -> WeatherSnapshot:
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

    return WeatherSnapshot(
        now=now,
        units=units,
        current_temp=current_temp,
        feels_like=feels_like,
        description=description,
        icon=icon,
        daily=daily,
    )

