from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from fractions import Fraction
from io import BytesIO
from typing import Any

import pytz
import requests
from PIL import Image, ImageDraw, ImageFont

from plugins.base_plugin.base_plugin import BasePlugin
from utils.app_utils import get_font, resolve_path

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - handled at runtime if missing
    genai = None
    genai_types = None

logger = logging.getLogger(__name__)


WEATHER_URL = (
    "https://api.openweathermap.org/data/3.0/onecall?"
    "lat={lat}&lon={long}&units={units}&exclude=minutely&appid={api_key}"
)


SPECTRA6_INSTRUCTIONS = (
    "Generate a flat, high-contrast children's book illustration using only black, white, red, green, blue, "
    "and yellow. The style must be poster-like with bold shapes, clean colour blocks, and no gradients or fine "
    "textures. Avoid subtle shading, soft edges, or photographic detail. This image will be displayed on a "
    "Spectra 6 e-ink panel with a limited colour gamut and slow refresh."
)


GEMINI_IMAGE_SIZES = {
    "1k": "1K",
    "2k": "2K",
    "4k": "4K",
}

GEMINI_IMAGE_CONFIG_UNSUPPORTED_MODELS = {
    # As of google-genai (v1beta), these image-generation models reject `image_config`.
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "models/gemini-2.5-flash-image",
    "models/gemini-3-pro-image-preview",
}

PROMPT_VERSION = 5
DEFAULT_CAT_DESCRIPTION = (
    "a larger-than-average (but not obese) orange-and-white cat (orange/ginger coat with white chest and paws; no other fur colours)"
)

LAYOUT_VERSION = 1
SIDEBAR_WIDTH_RATIO = 0.30


@dataclass(frozen=True)
class WeatherSnapshot:
    now: datetime
    units: str
    current_temp: float
    feels_like: float
    description: str
    icon: str
    daily: list[dict[str, Any]]


class DailyCatWeather(BasePlugin):
    def generate_settings_template(self):
        template_params = super().generate_settings_template()
        template_params["api_key"] = {
            "required": True,
            "service": "Gemini + OpenWeatherMap",
            "expected_key": "GEMINI_API_KEY / OPEN_WEATHER_MAP_SECRET",
        }
        return template_params

    def generate_image(self, settings, device_config):
        if genai is None or genai_types is None:
            raise RuntimeError(
                "Daily Cat Weather requires the 'google-genai' package. "
                "Run the InkyPi update script to install missing dependencies."
            )

        gemini_key = device_config.load_env_key("GEMINI_API_KEY")
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        owm_key = device_config.load_env_key("OPEN_WEATHER_MAP_SECRET")
        if not owm_key:
            raise RuntimeError("OPEN_WEATHER_MAP_SECRET is not configured.")

        lat = (settings.get("latitude") or "").strip()
        lon = (settings.get("longitude") or "").strip()
        if not lat or not lon:
            raise RuntimeError("Latitude and Longitude are required.")

        location_label = (settings.get("locationLabel") or "").strip()

        units = (settings.get("units") or "metric").strip().lower()
        if units not in {"metric", "imperial", "standard"}:
            raise RuntimeError("Units must be one of metric, imperial, or standard.")

        forecast_days = int(settings.get("forecastDays") or 3)
        forecast_days = max(1, min(5, forecast_days))

        model = (settings.get("imageModel") or "gemini-2.5-flash-image").strip()
        if not model.startswith("gemini-"):
            raise RuntimeError("Image model must be a Gemini model (gemini-*).")

        quality_key = (settings.get("quality") or "2k").strip().lower()
        image_size = GEMINI_IMAGE_SIZES.get(quality_key, "2K")

        cache_id = self._sanitize_cache_id(settings.get("cacheId") or "default")
        daily_refresh_time = self._parse_hhmm(settings.get("dailyRefreshTime") or "04:00")
        reroll_nonce = int(settings.get("rerollNonce") or 0)

        tz_str = device_config.get_config("timezone", default="UTC")
        tz = pytz.timezone(tz_str)
        now = datetime.now(tz)

        day_key = self._day_key(now, daily_refresh_time)
        cache_dir = self._cache_dir(device_config)
        os.makedirs(cache_dir, exist_ok=True)

        custom_prompt_day_key = (settings.get("customPromptDayKey") or "").strip()
        custom_prompt_raw = (settings.get("customPrompt") or "").strip()
        custom_prompt_enhanced = (settings.get("customPromptEnhanced") or "").strip()
        active_custom_prompt = (
            (custom_prompt_enhanced or custom_prompt_raw) if custom_prompt_day_key == day_key else ""
        )

        bg_path = os.path.join(cache_dir, f"bg_{cache_id}_{day_key}.png")
        meta_path = os.path.join(cache_dir, f"bg_{cache_id}_{day_key}.json")
        latest_link = os.path.join(cache_dir, f"latest_bg_{cache_id}.png")

        fingerprint_values = {
            "lat": lat,
            "lon": lon,
            "units": units,
            "model": model,
            "image_size": image_size,
            "daily_refresh_time": daily_refresh_time.strftime("%H:%M"),
            "reroll_nonce": reroll_nonce,
            "prompt_version": PROMPT_VERSION,
            "layout": "right_sidebar",
            "layout_version": LAYOUT_VERSION,
            "sidebar_ratio": SIDEBAR_WIDTH_RATIO,
        }
        if active_custom_prompt:
            fingerprint_values.update(
                {
                    "custom_prompt_day_key": custom_prompt_day_key,
                    "custom_prompt_hash": hashlib.sha256(active_custom_prompt.encode("utf-8")).hexdigest(),
                }
            )
        fingerprint = self._settings_fingerprint(fingerprint_values)

        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]

        width, height = dimensions
        sidebar_width = int(width * SIDEBAR_WIDTH_RATIO)
        sidebar_width = max(120, min(width - 100, sidebar_width))
        image_width = width - sidebar_width
        image_size_px = (image_width, height)
        sidebar_size_px = (sidebar_width, height)

        background = None
        meta = self._read_json(meta_path)
        if os.path.exists(bg_path) and meta and meta.get("fingerprint") == fingerprint:
            try:
                with Image.open(bg_path) as img:
                    background = img.convert("RGB")
            except Exception:
                logger.exception("Failed to load cached background; regenerating.")
                background = None
        if background is not None and background.size != image_size_px:
            background = None

        weather = None
        try:
            weather = self._fetch_weather_snapshot(owm_key, units, lat, lon, now)
        except Exception:
            logger.exception("Failed to fetch weather; continuing without overlay.")

        if background is None:
            aspect_hint = self._format_aspect_hint(image_width, height)
            if active_custom_prompt:
                prompt = self._build_custom_prompt(
                    weather,
                    active_custom_prompt,
                    reroll_nonce=reroll_nonce,
                    aspect_hint=aspect_hint,
                )
            else:
                prompt = self._build_prompt(
                    weather,
                    reroll_nonce=reroll_nonce,
                    cache_id=cache_id,
                    day_key=day_key,
                    aspect_hint=aspect_hint,
                )
            background = self._generate_gemini_background(
                api_key=gemini_key,
                prompt=prompt,
                model=model,
                image_size=image_size,
                aspect_ratio="9:16" if device_config.get_config("orientation") == "vertical" else "16:9",
            )
            background = self._trim_uniform_border(background)
            background = self._trim_internal_vertical_divider(background)
            background = self._cover_crop(background, image_size_px)
            background.save(bg_path)
            self._write_json(
                meta_path,
                {
                    "created_at": now.isoformat(),
                    "day_key": day_key,
                    "fingerprint": fingerprint,
                    "model": model,
                    "image_size": image_size,
                    "prompt": prompt,
                    "weather": (
                        {
                            "description": weather.description,
                            "temp": weather.current_temp,
                            "feels_like": weather.feels_like,
                        }
                        if weather
                        else {}
                    ),
                    "reroll_nonce": reroll_nonce,
                    "prompt_version": PROMPT_VERSION,
                    "custom_prompt": active_custom_prompt,
                    "custom_prompt_raw": custom_prompt_raw,
                    "custom_prompt_day_key": custom_prompt_day_key,
                    "layout": "right_sidebar",
                    "layout_version": LAYOUT_VERSION,
                    "sidebar_ratio": SIDEBAR_WIDTH_RATIO,
                    "sidebar_width": sidebar_width,
                },
            )
            try:
                background.save(latest_link)
            except Exception:
                logger.exception("Failed to update latest background pointer.")

        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(background, (0, 0))
        panel = self._render_weather_sidebar_panel(weather, tz, forecast_days, sidebar_size_px, location_label)
        canvas.paste(panel, (image_width, 0))
        return canvas

    @staticmethod
    def _cache_dir(device_config):
        return os.path.join(device_config.BASE_DIR, "..", "mock_display_output", "daily_cat_weather")

    @staticmethod
    def _sanitize_cache_id(value):
        value = (value or "").strip().lower()
        value = re.sub(r"[^a-z0-9_-]+", "-", value)
        value = re.sub(r"-{2,}", "-", value).strip("-")
        return value or "default"

    @staticmethod
    def _parse_hhmm(value):
        value = (value or "").strip()
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
        if not match:
            return time(4, 0)
        hour = int(match.group(1))
        minute = int(match.group(2))
        hour = max(0, min(23, hour))
        minute = max(0, min(59, minute))
        return time(hour, minute)

    @staticmethod
    def _day_key(now, refresh_time):
        if now.timetz().replace(tzinfo=None) < refresh_time:
            day = (now - timedelta(days=1)).date()
        else:
            day = now.date()
        return day.isoformat()

    @staticmethod
    def _settings_fingerprint(values):
        payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _fetch_weather_snapshot(self, api_key, units, lat, lon, now):
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

    def _build_prompt(self, weather, reroll_nonce=0, cache_id="default", day_key="", aspect_hint=""):
        base = (
            "Children's book illustration of an ambitious cat on a wholesome daily mission. "
            f"Main character: {DEFAULT_CAT_DESCRIPTION}. "
            "The cat's fur is strictly orange and white (no other fur colours). "
            "Keep it lighthearted and amusing, with a whimsical storybook vibe (not photorealistic). "
            "No text, no captions, no speech bubbles. "
        )

        if not weather:
            return (
                f"{base}"
                "Choose a fresh, creative mission for the cat and an interesting setting. "
                f"{SPECTRA6_INSTRUCTIONS}"
            )

        temp_c = self._to_celsius(weather.current_temp, weather.units)
        day_desc = weather.description.lower()
        activity, accessories, constraints = self._pick_activity(
            temp_c,
            day_desc,
            weather.daily,
            seed=f"{cache_id}|{day_key}|{reroll_nonce}|{day_desc}|{round(temp_c)}",
        )

        return (
            f"{base}"
            f"The scene matches today's weather: {weather.description}. "
            f"{constraints}"
            "Encourage creativity: pick an original setting and mission; avoid repeating the same scene across rerolls. "
            f"The cat is {activity}{accessories}. "
            f"Target aspect ratio: {aspect_hint}. "
            "Composition guidance: the final layout uses a dedicated weather sidebar on the right, so keep the main "
            "story action and characters centered and slightly left-of-center (avoid placing key details near the far "
            "right edge). Leave a little extra breathing room at the edges because the image will be center-cropped. "
            "Hard constraints: single scene only; do NOT create multiple panels, split-screen, triptych, frames, "
            "borders, dividers, callout boxes, badges, thermometers, gauges, charts, or any weather UI. "
            "Do not draw weather symbols/icons/emblems/logos (e.g. cloud icons, snowflake badges); depict the weather naturally in the scene. "
            "The illustration must fill the entire canvas edge-to-edge (no white margins, no empty borders). "
            "Absolutely no text, labels, or numbers anywhere in the illustration."
            f"{SPECTRA6_INSTRUCTIONS}"
        )

    def _build_custom_prompt(self, weather, user_prompt, reroll_nonce=0, aspect_hint=""):
        base = (
            "Children's book illustration of an ambitious cat on a wholesome daily mission. "
            f"Main character: {DEFAULT_CAT_DESCRIPTION}. "
            "The cat's fur is strictly orange and white (no other fur colours). "
            "Keep it lighthearted and amusing, with a whimsical storybook vibe (not photorealistic). "
            "No text, no captions, no speech bubbles. "
        )

        weather_line = ""
        temp_c = None
        if weather:
            temp_c = self._to_celsius(weather.current_temp, weather.units)
            weather_line = f"Today's weather: {weather.description}. "

        constraints = ""
        mild = temp_c is not None and temp_c >= 8
        if mild:
            prompt_lower = user_prompt.lower()
            if "fireplace" not in prompt_lower and "indoors" not in prompt_lower and "inside" not in prompt_lower:
                constraints = "Prefer an outdoor daylight setting (no fireplace / cozy indoor scenes). "

        return (
            f"{base}"
            f"{weather_line}"
            f"{constraints}"
            "Use the following scene idea as the main direction (you may add small visual details, but do not add new main subjects): "
            f"{user_prompt}. "
            "Full-bleed scene, no borders. "
            f"Target aspect ratio: {aspect_hint}. "
            "Composition guidance: the final layout uses a dedicated weather sidebar on the right, so keep the main "
            "story action and characters centered and slightly left-of-center (avoid placing key details near the far "
            "right edge). Leave a little extra breathing room at the edges because the image will be center-cropped. "
            "Hard constraints: single scene only; do NOT create multiple panels, split-screen, triptych, frames, "
            "borders, dividers, callout boxes, badges, thermometers, gauges, charts, or any weather UI. "
            "Do not draw standalone symbols/icons/emblems/logos/stamps/badges (e.g. cloud icons, snowflake badges); depict the weather naturally in the scene. "
            "The illustration must fill the entire canvas edge-to-edge (no white margins, no empty borders). "
            "Absolutely no text, labels, or numbers anywhere in the illustration."
            f"{SPECTRA6_INSTRUCTIONS}"
        )

    @staticmethod
    def _to_celsius(value, units):
        if units == "imperial":
            return (value - 32.0) * (5.0 / 9.0)
        if units == "standard":
            return value - 273.15
        return value

    @staticmethod
    def _pick_activity(temp_c, description, daily, seed=""):
        description = description or ""
        rng = random.Random(seed)

        pop = 0.0
        if daily and isinstance(daily, list) and daily[0]:
            try:
                pop = float(daily[0].get("pop") or 0.0)
            except (TypeError, ValueError):
                pop = 0.0

        accessories_rain = ""
        if pop >= 0.6:
            accessories_rain = " with a tiny umbrella and rain boots"

        constraints = ""
        if temp_c >= 8 and "snow" not in description:
            constraints = "Do not include a fireplace or indoor cozy scene; prefer an outdoor daylight setting. "

        if "snow" in description or "blizzard" in description:
            choices = [
                ("building a tiny snow fort and planning a daring expedition", " wearing a scarf and mittens"),
                ("sledding down a small hill like a brave explorer", " wearing a warm beanie and mittens"),
                ("helping build a snowcat sculpture for the neighborhood", " bundled up with earmuffs"),
            ]
            activity, accessories = rng.choice(choices)
            return activity, accessories, "Snowy conditions: bundle the cat up in winter gear. "
        if "rain" in description or "drizzle" in description or "storm" in description:
            choices = [
                ("puddle-jumping heroically on the way to deliver a letter", accessories_rain),
                ("sailing a leaf-boat flotilla down a little stream", accessories_rain),
                ("rescuing a tiny lost toy from the rain", accessories_rain),
            ]
            activity, accessories = rng.choice(choices)
            return activity, accessories, "Rainy conditions: give the cat rain gear. "
        if "fog" in description or "mist" in description or "haze" in description:
            choices = [
                ("navigating with a little compass like an explorer", " with an explorer hat"),
                ("following a treasure map through a misty garden maze", " carrying a tiny lantern"),
                ("acting as a brave 'lighthouse keeper' for lost friends", " holding a lantern"),
            ]
            activity, accessories = rng.choice(choices)
            return activity, accessories, "Low visibility: use a lantern/compass vibe and strong silhouettes. "

        if temp_c <= -10:
            choices = [
                ("checking on neighbors with a heroic winter patrol", " wrapped in a blanket like a cape"),
                ("delivering a tiny thermos of cocoa to a friend", " wearing a puffy jacket and scarf"),
                ("building a windbreak fort and planting a little flag on the hilltop", " wearing a scarf"),
            ]
            activity, accessories = rng.choice(choices)
            return activity, accessories, "Very cold: show winter clothing and cold air. "
        if temp_c <= 0:
            choices = [
                ("ice skating very carefully while fully bundled up", " wearing earmuffs and a puffy jacket"),
                ("making a 'hot cocoa delivery' run with a little satchel", " wearing a scarf"),
                ("trying snowshoe steps with homemade paw 'skis'", " bundled up with a beanie"),
            ]
            activity, accessories = rng.choice(choices)
            return activity, accessories, "Cold: bundle up the cat in warm clothing. "
        if temp_c >= 28:
            choices = [
                ("leading a garden watering mission and chasing sunbeams", " wearing sunglasses"),
                ("running a lemonade stand for animal friends", " wearing a sunhat"),
                ("building a tiny shaded 'cool-down station' with a fan", " holding a cold drink"),
            ]
            activity, accessories = rng.choice(choices)
            return activity, accessories, "Hot weather: bright sunlight, shade, and cool drinks. "

        choices = [
            ("going on a cheerful neighborhood adventure to help a friend", ""),
            ("building a tiny birdhouse workshop and doing careful 'construction'", " with a little tool belt"),
            ("hosting a mini picnic and sharing snacks with animal friends", " carrying a picnic basket"),
            ("painting a cheerful mural on a fence with a little brush", " holding a paintbrush"),
            ("collecting leaves and flowers for a 'nature museum' exhibit", " carrying a little basket"),
            ("flying a kite on a breezy hill like a proud adventurer", " holding a kite string"),
            ("setting up a sidewalk 'library cart' and handing out stories", " pushing a tiny cart"),
        ]
        activity, accessories = rng.choice(choices)
        return activity, accessories, constraints

    def _render_weather_overlay(self, weather, tz, forecast_days, dimensions):
        width, height = dimensions
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        pad = max(8, int(width * 0.018))
        badge_h = max(64, int(height * 0.14))
        badge_w = max(220, int(width * 0.30))
        bar_h = max(92, int(height * 0.18))

        badge_box = (pad, pad, pad + badge_w, pad + badge_h)
        bar_box = (pad, height - pad - bar_h, width - pad, height - pad)

        self._rounded(draw, badge_box, fill=(255, 255, 255, 220), radius=14)
        self._rounded(draw, bar_box, fill=(255, 255, 255, 220), radius=16)

        icon_path = self._weather_icon_path(weather.icon)
        icon_img = self._load_icon(icon_path, size=int(badge_h * 0.75))

        title_font = self._font("Jost", int(badge_h * 0.22))
        temp_font = self._font("Jost", int(badge_h * 0.42), bold=True)
        small_font = self._font("Jost", int(badge_h * 0.18))

        temp_value = round(weather.current_temp)
        feels_value = round(weather.feels_like)
        temp_unit = "°C" if weather.units == "metric" else ("°F" if weather.units == "imperial" else "K")

        x0, y0, x1, y1 = badge_box
        icon_x = x0 + pad
        icon_y = y0 + int((badge_h - icon_img.height) / 2)
        overlay.alpha_composite(icon_img, (icon_x, icon_y))

        text_x = icon_x + icon_img.width + int(pad * 0.8)
        draw.text((text_x, y0 + int(pad * 0.2)), "Now", fill=(0, 0, 0, 255), font=title_font)
        draw.text(
            (text_x, y0 + int(badge_h * 0.28)),
            f"{temp_value}{temp_unit}",
            fill=(0, 0, 0, 255),
            font=temp_font,
        )
        draw.text(
            (text_x, y0 + int(badge_h * 0.75)),
            f"Feels {feels_value}{temp_unit}",
            fill=(0, 0, 0, 200),
            font=small_font,
        )

        self._draw_forecast_bar(draw, overlay, weather, tz, forecast_days, bar_box, pad)

        return overlay

    def _draw_forecast_bar(self, draw, overlay, weather, tz, forecast_days, bar_box, pad):
        x0, y0, x1, y1 = bar_box
        width = x1 - x0
        height = y1 - y0

        day_font = self._font("Jost", int(height * 0.18), bold=True)
        temp_font = self._font("Jost", int(height * 0.17))
        pop_font = self._font("Jost", int(height * 0.15))

        daily = weather.daily[1 : 1 + forecast_days] if weather.daily else []
        cols = max(1, forecast_days)
        col_w = int((width - pad * 2) / cols)
        base_y = y0 + int(pad * 0.35)

        for idx in range(cols):
            if idx >= len(daily):
                continue
            day = daily[idx] or {}
            dt = datetime.fromtimestamp(int(day.get("dt", 0)), tz=timezone.utc).astimezone(tz)
            label = dt.strftime("%a")

            icon_code = "01d"
            try:
                icon_code = str((day.get("weather") or [{}])[0].get("icon") or "01d").replace("n", "d")
            except Exception:
                icon_code = "01d"

            temps = day.get("temp") or {}
            high = round(float(temps.get("max") or 0.0))
            low = round(float(temps.get("min") or 0.0))
            pop = 0
            try:
                pop = int(round(float(day.get("pop") or 0.0) * 100))
            except (TypeError, ValueError):
                pop = 0

            col_x = x0 + pad + idx * col_w
            icon_img = self._load_icon(self._weather_icon_path(icon_code), size=int(height * 0.35))
            overlay.alpha_composite(icon_img, (col_x, base_y + int(height * 0.05)))

            text_x = col_x + icon_img.width + int(pad * 0.5)
            draw.text((text_x, base_y), label, fill=(0, 0, 0, 255), font=day_font)
            draw.text((text_x, base_y + int(height * 0.24)), f"H {high}  L {low}", fill=(0, 0, 0, 230), font=temp_font)
            draw.text((text_x, base_y + int(height * 0.43)), f"Precip {pop}%", fill=(0, 0, 0, 200), font=pop_font)

    @staticmethod
    def _cover_crop(image, target_size):
        target_w, target_h = target_size
        if target_w <= 0 or target_h <= 0:
            raise ValueError("Invalid target size for cover crop.")

        img = image.convert("RGB")
        src_w, src_h = img.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError("Invalid source image size.")

        scale = max(target_w / src_w, target_h / src_h)
        resized_w = max(target_w, int(round(src_w * scale)))
        resized_h = max(target_h, int(round(src_h * scale)))
        img = img.resize((resized_w, resized_h), Image.LANCZOS)

        left = max(0, int((resized_w - target_w) / 2))
        top = max(0, int((resized_h - target_h) / 2))
        return img.crop((left, top, left + target_w, top + target_h))

    @staticmethod
    def _trim_uniform_border(image, bg=(255, 255, 255), tolerance=14, min_coverage=0.985, max_crop_ratio=0.22):
        """Trim uniform edge borders (e.g., white margins) while avoiding aggressive crops."""
        img = image.convert("RGB")
        w, h = img.size
        if w < 10 or h < 10:
            return img

        px = img.load()
        max_left = int(w * max_crop_ratio)
        max_right = int(w * max_crop_ratio)
        max_top = int(h * max_crop_ratio)
        max_bottom = int(h * max_crop_ratio)

        def near_bg(rgb):
            return (
                abs(rgb[0] - bg[0]) <= tolerance
                and abs(rgb[1] - bg[1]) <= tolerance
                and abs(rgb[2] - bg[2]) <= tolerance
            )

        def edge_coverage_left(x):
            hits = 0
            for yy in range(h):
                if near_bg(px[x, yy]):
                    hits += 1
            return hits / h

        def edge_coverage_right(x):
            hits = 0
            for yy in range(h):
                if near_bg(px[x, yy]):
                    hits += 1
            return hits / h

        def edge_coverage_top(y):
            hits = 0
            for xx in range(w):
                if near_bg(px[xx, y]):
                    hits += 1
            return hits / w

        def edge_coverage_bottom(y):
            hits = 0
            for xx in range(w):
                if near_bg(px[xx, y]):
                    hits += 1
            return hits / w

        left = 0
        right = w
        top = 0
        bottom = h

        # Iterate until edges are no longer mostly background or we hit crop limits.
        while left < max_left and edge_coverage_left(left) >= min_coverage:
            left += 1
        while (w - right) < max_right and edge_coverage_right(right - 1) >= min_coverage:
            right -= 1
        while top < max_top and edge_coverage_top(top) >= min_coverage:
            top += 1
        while (h - bottom) < max_bottom and edge_coverage_bottom(bottom - 1) >= min_coverage:
            bottom -= 1

        if right - left < int(w * 0.55) or bottom - top < int(h * 0.55):
            return img
        if left == 0 and right == w and top == 0 and bottom == h:
            return img
        return img.crop((left, top, right, bottom))

    @staticmethod
    def _trim_internal_vertical_divider(image, tolerance=18, min_coverage=0.93):
        """Trim right-side panel artifacts separated by a strong vertical divider line."""
        img = image.convert("RGB")
        w, h = img.size
        if w < 80 or h < 80:
            return img

        px = img.load()

        def near_white(rgb):
            return rgb[0] >= 255 - tolerance and rgb[1] >= 255 - tolerance and rgb[2] >= 255 - tolerance

        def near_black(rgb):
            return rgb[0] <= tolerance and rgb[1] <= tolerance and rgb[2] <= tolerance

        def divider_coverage(x):
            hits = 0
            for yy in range(h):
                c = px[x, yy]
                if near_white(c) or near_black(c):
                    hits += 1
            return hits / h

        search_left = int(w * 0.35)
        search_right = int(w * 0.92)

        # scan from right to left to find the first strong divider
        divider_x = None
        for x in range(search_right, search_left, -1):
            if divider_coverage(x) >= min_coverage:
                divider_x = x
                break

        if divider_x is None:
            return img

        # expand to a run of divider-like columns
        run_start = divider_x
        while run_start > search_left and divider_coverage(run_start - 1) >= min_coverage:
            run_start -= 1

        # Crop everything to the left of the divider run.
        new_right = max(int(w * 0.55), run_start)
        if new_right >= w:
            return img
        return img.crop((0, 0, new_right, h))

    @staticmethod
    def _format_aspect_hint(width, height):
        if width <= 0 or height <= 0:
            return "unknown"
        ratio = width / height
        if ratio < 0.9:
            return "portrait, close to square"
        if ratio > 1.25:
            return "landscape, wide"
        return "landscape, near-square"

    def _render_weather_sidebar_panel(self, weather, tz, forecast_days, sidebar_size, location_label=""):
        panel_w, panel_h = sidebar_size
        panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
        draw = ImageDraw.Draw(panel)

        def _text_bbox(text, font):
            try:
                return draw.textbbox((0, 0), text, font=font)
            except Exception:
                try:
                    box = font.getbbox(text)
                    return (0, 0, box[2] - box[0], box[3] - box[1])
                except Exception:
                    w, h = font.getsize(text)  # type: ignore[attr-defined]
                    return (0, 0, w, h)

        def _text_size(text, font):
            b = _text_bbox(text, font)
            return (b[2] - b[0], b[3] - b[1])

        def _wrap_text(text, font, max_width, max_lines=2):
            words = (text or "").strip().split()
            if not words:
                return []

            lines = []
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                w, _ = _text_size(candidate, font)
                if w <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)

            if max_lines and len(lines) > max_lines:
                lines = lines[:max_lines]
                while True:
                    w, _ = _text_size(lines[-1] + "…", font)
                    if w <= max_width or len(lines[-1]) <= 1:
                        break
                    lines[-1] = lines[-1].rsplit(" ", 1)[0]
                lines[-1] = lines[-1] + "…"
            return lines

        def _line_height(font):
            try:
                ascent, descent = font.getmetrics()
                return max(1, int(ascent + descent))
            except Exception:
                return max(1, _text_size("Ag", font)[1])

        draw.line((0, 0, 0, panel_h), fill=(0, 0, 0))
        pad = max(10, int(panel_w * 0.06))
        gap = max(6, int(pad * 0.5))

        title_font = self._font("Jost", max(14, int(panel_w * 0.085)), bold=True)
        base_temp_font_size = max(18, int(panel_w * 0.16))
        small_font = self._font("Jost", max(12, int(panel_w * 0.065)))
        location_font = self._font("Jost", max(12, int(panel_w * 0.070)))
        text_secondary = (0, 0, 0)
        icon_accent = (0, 0, 0, 255)

        y = pad
        if not weather:
            draw.text((pad, y), "Weather", fill=(0, 0, 0), font=title_font)
            draw.text((pad, y + int(pad * 1.4)), "Unavailable", fill=(0, 0, 0), font=small_font)
            return panel

        if location_label:
            draw.text((pad, y), location_label, fill=text_secondary, font=location_font)
            y += _line_height(location_font) + int(gap * 0.8)

        temp_value = round(weather.current_temp)
        feels_value = round(weather.feels_like)
        temp_unit = "°C" if weather.units == "metric" else ("°F" if weather.units == "imperial" else "K")
        desc = (weather.description or "").strip().capitalize()

        icon_size = max(40, int(panel_w * 0.22))
        icon = self._simple_weather_icon(weather.icon, size=icon_size).convert("RGBA")

        icon_x = pad
        icon_y = y
        panel.paste(icon, (icon_x, icon_y), icon)

        text_x = icon_x + icon_size + gap
        max_text_w = panel_w - pad - text_x
        max_text_w = max(40, max_text_w)

        def _fit_font(text, max_width, start_size, min_size=12):
            size = start_size
            while size > min_size:
                font = self._font("Jost", size, bold=True)
                if _text_size(text, font)[0] <= max_width:
                    return font
                size -= 2
            return self._font("Jost", min_size, bold=True)

        small_lh = _line_height(small_font)
        line_gap = max(6, int(small_lh * 0.35))
        y_cursor = icon_y

        temp_font = _fit_font(f"{temp_value}{temp_unit}", max_text_w, base_temp_font_size, min_size=14)
        temp_lh = _line_height(temp_font)
        temp_y = icon_y + max(0, int((icon_size - temp_lh) / 2))
        draw.text((text_x, temp_y), f"{temp_value}{temp_unit}", fill=(0, 0, 0), font=temp_font)

        y_cursor = icon_y + max(icon_size, (temp_y - icon_y) + temp_lh) + int(gap * 0.7)

        if desc:
            for line in _wrap_text(desc, small_font, max_text_w, max_lines=2):
                draw.text((text_x, y_cursor), line, fill=(0, 0, 0), font=small_font)
                y_cursor += small_lh + int(line_gap * 0.9)

        feels_line = f"Feels like {feels_value}{temp_unit}"
        draw.text((text_x, y_cursor), feels_line, fill=text_secondary, font=small_font)
        y_cursor += small_lh + line_gap

        header_h = max(icon_size, y_cursor - icon_y)
        divider_y = icon_y + header_h + int(pad * 0.7)
        draw.line((pad, divider_y, panel_w - pad, divider_y), fill=(0, 0, 0))
        y = divider_y + int(pad * 0.7)

        daily = weather.daily[1 : 1 + forecast_days] if weather.daily else []
        if not daily:
            draw.text((pad, y), "No forecast", fill=(0, 0, 0), font=small_font)
            return panel

        remaining_h = panel_h - y - pad
        row_h = max(64, int(remaining_h / max(1, forecast_days)))
        row_icon = max(28, int(row_h * 0.55))
        row_day_font = self._font("Jost", max(13, int(row_h * 0.24)), bold=True)
        base_row_temp_font_size = max(12, int(row_h * 0.20))
        row_temp_font = self._font("Jost", base_row_temp_font_size)
        row_lh = max(_line_height(row_temp_font), _line_height(row_day_font))

        precip_col_w = max(56, int(panel_w * 0.30))
        precip_x0 = panel_w - pad - precip_col_w
        precip_x1 = panel_w - pad
        temp_x0 = pad + row_icon + gap
        temp_x1 = precip_x0 - gap
        temp_x1 = max(temp_x0 + 40, temp_x1)

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_w_max = max((_text_size(d, row_day_font)[0] for d in day_labels), default=34)
        day_col_w = max(day_w_max + gap, 34)

        def _fit_row_font(text, max_width, start_size, min_size=10):
            size = start_size
            while size > min_size:
                font = self._font("Jost", size)
                if _text_size(text, font)[0] <= max_width:
                    return font
                size -= 1
            return self._font("Jost", min_size)

        def _draw_droplet_icon(x, y_mid, size, color):
            r = max(1, int(size * 0.22))
            body_h = max(2, int(size * 0.58))
            top_y = y_mid - int(body_h / 2)
            bottom_y = top_y + body_h
            cx = x + int(size / 2)
            draw.polygon([(cx, top_y), (x + size, bottom_y - r), (x, bottom_y - r)], fill=color)
            draw.ellipse((x, bottom_y - 2 * r, x + 2 * r, bottom_y), fill=color)
            draw.ellipse((x + size - 2 * r, bottom_y - 2 * r, x + size, bottom_y), fill=color)

        for idx, day in enumerate(daily[:forecast_days]):
            row_y0 = y + idx * row_h
            row_y1 = min(panel_h - pad, row_y0 + row_h)
            if row_y0 >= panel_h - pad:
                break

            if idx:
                draw.line((pad, row_y0, panel_w - pad, row_y0), fill=(0, 0, 0))

            dt = datetime.fromtimestamp(int((day or {}).get("dt", 0)), tz=timezone.utc).astimezone(tz)
            label = dt.strftime("%a")

            icon_code = "01d"
            try:
                icon_code = str(((day or {}).get("weather") or [{}])[0].get("icon") or "01d").replace("n", "d")
            except Exception:
                icon_code = "01d"

            temps = (day or {}).get("temp") or {}
            high = round(float(temps.get("max") or 0.0))
            low = round(float(temps.get("min") or 0.0))
            pop = 0
            try:
                pop = int(round(float((day or {}).get("pop") or 0.0) * 100))
            except (TypeError, ValueError):
                pop = 0

            icon_img = self._simple_weather_icon(icon_code, size=row_icon).convert("RGBA")
            icon_y = row_y0 + int((row_h - row_icon) / 2)
            panel.paste(icon_img, (pad, icon_y), icon_img)

            row_text_y = row_y0 + int((row_h - row_lh) / 2)
            draw.text((temp_x0, row_text_y), label, fill=(0, 0, 0), font=row_day_font)

            temps_area_left = temp_x0 + day_col_w
            temps_area_right = temp_x1
            temps_max_w = max(10, temps_area_right - temps_area_left)
            temps_line = f"{high}°/{low}°"
            if _text_size(temps_line, row_temp_font)[0] > temps_max_w:
                temps_line = f"{high}/{low}"
            temps_font = _fit_row_font(temps_line, temps_max_w, base_row_temp_font_size, min_size=10)
            draw.text((temps_area_left, row_text_y), temps_line, fill=(0, 0, 0), font=temps_font)

            precip_line = f"{pop}%"
            pw, _ = _text_size(precip_line, temps_font)
            drop_size = max(10, int(row_lh * 0.58))
            group_gap = max(3, int(gap * 0.25))
            group_w = drop_size + group_gap + pw
            group_x0 = max(precip_x0, precip_x1 - group_w)
            drop_x = group_x0
            precip_x = drop_x + drop_size + group_gap
            _draw_droplet_icon(drop_x, row_text_y + int(row_lh / 2), drop_size, icon_accent)
            draw.text((precip_x, row_text_y), precip_line, fill=(0, 0, 0), font=temps_font)

        return panel

    @staticmethod
    def _simple_weather_icon(icon_code, size):
        code = str(icon_code or "01d").lower()
        key = code[:2]

        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        stroke = (35, 35, 35, 255)
        fill_cloud = (245, 245, 245, 255)
        fill_sun = (255, 220, 120, 255)
        fill_rain = (60, 130, 220, 255)
        width = max(2, int(size * 0.06))

        def _cloud(x0, y0, w, h, outline=stroke, fill=fill_cloud):
            r = int(min(w, h) * 0.35)
            base_y = y0 + int(h * 0.55)
            draw.rounded_rectangle((x0, base_y - r, x0 + w, y0 + h), radius=r, fill=fill, outline=outline, width=width)
            draw.ellipse((x0 + int(w * 0.05), y0 + int(h * 0.20), x0 + int(w * 0.45), y0 + int(h * 0.75)), fill=fill, outline=outline, width=width)
            draw.ellipse((x0 + int(w * 0.30), y0 + int(h * 0.05), x0 + int(w * 0.75), y0 + int(h * 0.80)), fill=fill, outline=outline, width=width)
            draw.ellipse((x0 + int(w * 0.60), y0 + int(h * 0.25), x0 + int(w * 0.95), y0 + int(h * 0.78)), fill=fill, outline=outline, width=width)

        def _sun(cx, cy, r):
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill_sun, outline=stroke, width=width)
            ray_len = int(r * 0.65)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
                x1 = cx + int(dx * (r + 2))
                y1 = cy + int(dy * (r + 2))
                x2 = cx + int(dx * (r + ray_len))
                y2 = cy + int(dy * (r + ray_len))
                draw.line((x1, y1, x2, y2), fill=stroke, width=width)

        def _rain(x0, y0, w, h, drops=3):
            for i in range(drops):
                x = x0 + int((i + 0.5) * w / drops)
                draw.line((x, y0, x - int(width * 0.4), y0 + h), fill=fill_rain, width=width)

        def _snow(x0, y0, w, h):
            for i in range(2):
                x = x0 + int((i + 0.6) * w / 2)
                y = y0 + int(h * 0.35) + i * int(h * 0.25)
                r = int(width * 1.2)
                draw.line((x - r, y, x + r, y), fill=fill_rain, width=max(1, int(width * 0.6)))
                draw.line((x, y - r, x, y + r), fill=fill_rain, width=max(1, int(width * 0.6)))
                draw.line((x - r, y - r, x + r, y + r), fill=fill_rain, width=max(1, int(width * 0.6)))
                draw.line((x - r, y + r, x + r, y - r), fill=fill_rain, width=max(1, int(width * 0.6)))

        def _mist(x0, y0, w, h):
            line_w = max(1, int(width * 0.8))
            for i in range(3):
                yy = y0 + int((i + 1) * h / 4)
                draw.line((x0, yy, x0 + w, yy), fill=(120, 120, 120, 255), width=line_w)

        if key == "01":
            _sun(int(size * 0.50), int(size * 0.50), int(size * 0.22))
            return img

        if key == "02":
            _sun(int(size * 0.40), int(size * 0.38), int(size * 0.18))
            _cloud(int(size * 0.18), int(size * 0.30), int(size * 0.70), int(size * 0.55))
            return img

        if key in {"03", "04"}:
            _cloud(int(size * 0.12), int(size * 0.28), int(size * 0.76), int(size * 0.58))
            return img

        if key in {"09", "10"}:
            _cloud(int(size * 0.12), int(size * 0.22), int(size * 0.76), int(size * 0.55))
            _rain(int(size * 0.26), int(size * 0.62), int(size * 0.46), int(size * 0.25), drops=3)
            return img

        if key == "11":
            _cloud(int(size * 0.12), int(size * 0.22), int(size * 0.76), int(size * 0.55))
            bolt = [
                (int(size * 0.55), int(size * 0.58)),
                (int(size * 0.42), int(size * 0.92)),
                (int(size * 0.58), int(size * 0.92)),
                (int(size * 0.48), int(size * 1.08)),
            ]
            draw.polygon(bolt, fill=(255, 210, 80, 255))
            draw.line(bolt + [bolt[0]], fill=stroke, width=max(1, int(width * 0.6)))
            return img

        if key == "13":
            _cloud(int(size * 0.12), int(size * 0.22), int(size * 0.76), int(size * 0.55))
            _snow(int(size * 0.22), int(size * 0.62), int(size * 0.56), int(size * 0.28))
            return img

        if key == "50":
            _mist(int(size * 0.18), int(size * 0.28), int(size * 0.70), int(size * 0.55))
            return img

        _cloud(int(size * 0.12), int(size * 0.28), int(size * 0.76), int(size * 0.58))
        return img

    @staticmethod
    def _weather_icon_path(icon_code):
        return resolve_path(os.path.join("plugins", "weather", "icons", f"{icon_code}.png"))

    @staticmethod
    def _load_icon(path, size):
        try:
            with Image.open(path) as img:
                img = img.convert("RGBA")
                img = img.resize((size, size), Image.LANCZOS)
                return img
        except Exception:
            fallback = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            return fallback

    @staticmethod
    def _rounded(draw, box, fill, radius):
        try:
            draw.rounded_rectangle(box, radius=radius, fill=fill)
        except Exception:
            draw.rectangle(box, fill=fill)

    @staticmethod
    def _font(family, size, bold=False):
        font = get_font(family, font_size=size, font_weight="bold" if bold else "normal")
        if font is None:
            return ImageFont.load_default()
        return font

    @staticmethod
    def _composite(background, overlay):
        base = background.convert("RGBA")
        base.alpha_composite(overlay)
        return base.convert("RGB")

    @staticmethod
    def _read_json(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    @staticmethod
    def _write_json(path, payload):
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except Exception:
            logger.exception("Failed to write cache metadata: %s", path)

    @staticmethod
    def _generate_gemini_background(api_key, prompt, model, image_size, aspect_ratio):
        def _normalize_model_name(name):
            name = (name or "").strip()
            return name

        def _decode_image_bytes(image_bytes):
            try:
                return Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception as exc:
                logger.exception("Failed to decode Gemini image bytes: %s", exc)
                raise RuntimeError("Gemini image decoding failure, please check logs.") from exc

        def _generate_content_minimal(client):
            # Use a minimal config because some models reject image_config.
            # We guide aspect ratio/size via prompt and then resize to panel resolution.
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )

        def _extract_image_bytes_from_generate_content(response):
            try:
                candidate = response.candidates[0]
                return next(
                    (
                        part.inline_data.data
                        for part in candidate.content.parts
                        if getattr(part, "inline_data", None)
                    ),
                    None,
                )
            except Exception:
                logger.exception("Unexpected Gemini response format.")
                return None

        def _looks_like_invalid_argument(exc):
            message = str(exc).upper()
            return "INVALID_ARGUMENT" in message or "400" in message

        try:
            client = genai.Client(api_key=api_key)

            orientation_hint = "landscape" if aspect_ratio == "16:9" else "portrait"
            prompt = (
                f"{prompt}\n\n"
                f"Output format: {orientation_hint} orientation, full-bleed, no borders. No text or symbols."
            )

            normalized_model = _normalize_model_name(model)
            if normalized_model in GEMINI_IMAGE_CONFIG_UNSUPPORTED_MODELS:
                response = _generate_content_minimal(client)
            else:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=genai_types.ImageConfig(
                                aspect_ratio=aspect_ratio,
                                image_size=image_size,
                            ),
                        ),
                    )
                except Exception as exc:
                    logger.warning("Gemini image_config rejected; retrying without image_config: %s", exc)
                    if not _looks_like_invalid_argument(exc):
                        raise

                    try:
                        response = _generate_content_minimal(client)
                    except Exception as exc2:
                        logger.exception("Gemini generate_content retry failed (minimal IMAGE): %s", exc2)
                        # Final retry: include TEXT modality in case the model requires it.
                        response = client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=genai_types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
                        )
        except Exception as exc:
            logger.exception("Gemini request failed: %s", exc)
            raise RuntimeError(f"Gemini image request failure: {exc}") from exc

        image_bytes = None
        image_bytes = _extract_image_bytes_from_generate_content(response)

        if not image_bytes:
            raise RuntimeError("Gemini returned no image data.")

        return _decode_image_bytes(image_bytes)
