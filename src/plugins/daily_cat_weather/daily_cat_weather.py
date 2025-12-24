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

PROMPT_VERSION = 3
DEFAULT_CAT_DESCRIPTION = (
    "a larger-than-average (but not obese) orange-and-white cat (ginger tabby with a white chest and paws)"
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
        panel = self._render_weather_sidebar_panel(weather, tz, forecast_days, sidebar_size_px)
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
            f"The scene matches today's weather: {weather.description} (about {temp_c:.0f}°C). "
            f"{constraints}"
            "Encourage creativity: pick an original setting and mission; avoid repeating the same scene across rerolls. "
            f"The cat is {activity}{accessories}. "
            f"Variant id: {reroll_nonce}. "
            f"Target aspect ratio: {aspect_hint}. "
            "Composition guidance: the final layout uses a dedicated weather sidebar on the right, so keep the main "
            "story action and characters centered and slightly left-of-center (avoid placing key details near the far "
            "right edge). Leave a little extra breathing room at the edges because the image will be center-cropped. "
            f"{SPECTRA6_INSTRUCTIONS}"
        )

    def _build_custom_prompt(self, weather, user_prompt, reroll_nonce=0, aspect_hint=""):
        base = (
            "Children's book illustration of an ambitious cat on a wholesome daily mission. "
            f"Main character: {DEFAULT_CAT_DESCRIPTION}. "
            "Keep it lighthearted and amusing, with a whimsical storybook vibe (not photorealistic). "
            "No text, no captions, no speech bubbles. "
        )

        weather_line = ""
        temp_c = None
        if weather:
            temp_c = self._to_celsius(weather.current_temp, weather.units)
            weather_line = f"Today's weather: {weather.description} (about {temp_c:.0f}°C). "

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
            f"Variant id: {reroll_nonce}. "
            f"Target aspect ratio: {aspect_hint}. "
            "Composition guidance: the final layout uses a dedicated weather sidebar on the right, so keep the main "
            "story action and characters centered and slightly left-of-center (avoid placing key details near the far "
            "right edge). Leave a little extra breathing room at the edges because the image will be center-cropped. "
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
    def _format_aspect_hint(width, height):
        if width <= 0 or height <= 0:
            return "unknown"
        ratio = Fraction(width, height).limit_denominator(12)
        return f"{ratio.numerator}:{ratio.denominator} (~{width/height:.2f}:1)"

    def _render_weather_sidebar_panel(self, weather, tz, forecast_days, sidebar_size):
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

        draw.line((0, 0, 0, panel_h), fill=(0, 0, 0))
        pad = max(10, int(panel_w * 0.06))
        gap = max(6, int(pad * 0.5))

        title_font = self._font("Jost", max(14, int(panel_w * 0.085)), bold=True)
        temp_font = self._font("Jost", max(18, int(panel_w * 0.18)), bold=True)
        small_font = self._font("Jost", max(12, int(panel_w * 0.065)))

        y = pad
        if not weather:
            draw.text((pad, y), "Weather", fill=(0, 0, 0), font=title_font)
            draw.text((pad, y + int(pad * 1.4)), "Unavailable", fill=(0, 0, 0), font=small_font)
            return panel

        temp_value = round(weather.current_temp)
        feels_value = round(weather.feels_like)
        temp_unit = "°C" if weather.units == "metric" else ("°F" if weather.units == "imperial" else "K")
        desc = (weather.description or "").strip().capitalize()

        icon_size = max(40, int(panel_w * 0.22))
        icon = self._load_icon(self._weather_icon_path(weather.icon), size=icon_size).convert("RGB")

        icon_x = pad
        icon_y = y
        panel.paste(icon, (icon_x, icon_y))

        text_x = icon_x + icon_size + gap
        max_text_w = panel_w - pad - text_x

        line_gap = max(3, int(small_font.size * 0.35)) if hasattr(small_font, "size") else 4
        y_cursor = icon_y

        for line in _wrap_text("Now", small_font, max_text_w, max_lines=1):
            draw.text((text_x, y_cursor), line, fill=(0, 0, 0), font=small_font)
            y_cursor += _text_size(line, small_font)[1] + line_gap

        for line in _wrap_text(f"{temp_value}{temp_unit}", temp_font, max_text_w, max_lines=1):
            draw.text((text_x, y_cursor), line, fill=(0, 0, 0), font=temp_font)
            y_cursor += _text_size(line, temp_font)[1] + line_gap

        feels_line = f"Feels {feels_value}{temp_unit}"
        for line in _wrap_text(feels_line, small_font, max_text_w, max_lines=1):
            draw.text((text_x, y_cursor), line, fill=(0, 0, 0), font=small_font)
            y_cursor += _text_size(line, small_font)[1] + line_gap

        if desc:
            for line in _wrap_text(desc, small_font, max_text_w, max_lines=2):
                draw.text((text_x, y_cursor), line, fill=(0, 0, 0), font=small_font)
                y_cursor += _text_size(line, small_font)[1] + line_gap

        header_h = max(icon_size, y_cursor - icon_y)
        y = icon_y + header_h + pad

        draw.text((pad, y), f"Next {forecast_days} days", fill=(0, 0, 0), font=title_font)
        y += _text_size(f"Next {forecast_days} days", title_font)[1] + int(pad * 0.6)

        daily = weather.daily[1 : 1 + forecast_days] if weather.daily else []
        if not daily:
            draw.text((pad, y), "No forecast", fill=(0, 0, 0), font=small_font)
            return panel

        remaining_h = panel_h - y - pad
        row_h = max(70, int(remaining_h / max(1, forecast_days)))
        row_pad = max(6, int(row_h * 0.12))
        row_icon = max(30, int(row_h * 0.50))
        row_day_font = self._font("Jost", max(14, int(row_h * 0.22)), bold=True)
        row_temp_font = self._font("Jost", max(12, int(row_h * 0.20)))

        precip_col_w = max(54, int(panel_w * 0.30))
        precip_x0 = panel_w - pad - precip_col_w
        precip_x1 = panel_w - pad
        temp_x0 = pad + row_icon + gap
        temp_x1 = precip_x0 - gap

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

            icon_img = self._load_icon(self._weather_icon_path(icon_code), size=row_icon).convert("RGB")
            icon_y = row_y0 + int((row_h - row_icon) / 2)
            panel.paste(icon_img, (pad, icon_y))

            day_y = row_y0 + row_pad
            temp_y = day_y + _text_size(label, row_day_font)[1] + int(row_pad * 0.35)
            temp_y = min(temp_y, row_y1 - row_pad - _text_size("0° / 0°", row_temp_font)[1])

            draw.text((temp_x0, day_y), label, fill=(0, 0, 0), font=row_day_font)

            temps_line = f"{high}° / {low}°"
            if _text_size(temps_line, row_temp_font)[0] > (temp_x1 - temp_x0):
                temps_line = f"H{high} L{low}"
            draw.text((temp_x0, temp_y), temps_line, fill=(0, 0, 0), font=row_temp_font)

            precip_line = f"{pop}%"
            pw, ph = _text_size(precip_line, row_temp_font)
            precip_y = temp_y
            precip_x = precip_x1 - pw
            draw.text((precip_x, precip_y), precip_line, fill=(0, 0, 0), font=row_temp_font)

        return panel

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
                f"Output format: {orientation_hint} {aspect_ratio} aspect ratio, full-bleed, no borders."
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
