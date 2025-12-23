from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from io import BytesIO
from typing import Any

import pytz
import requests
from PIL import Image, ImageDraw, ImageFont

from plugins.base_plugin.base_plugin import BasePlugin
from utils.app_utils import get_font, resolve_path
from utils.image_utils import resize_image

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

PROMPT_VERSION = 2
DEFAULT_CAT_DESCRIPTION = (
    "a larger-than-average (but not obese) orange-and-white cat (ginger tabby with a white chest and paws)"
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

        bg_path = os.path.join(cache_dir, f"bg_{cache_id}_{day_key}.png")
        meta_path = os.path.join(cache_dir, f"bg_{cache_id}_{day_key}.json")
        latest_link = os.path.join(cache_dir, f"latest_bg_{cache_id}.png")

        fingerprint = self._settings_fingerprint(
            {
                "lat": lat,
                "lon": lon,
                "units": units,
                "model": model,
                "image_size": image_size,
                "daily_refresh_time": daily_refresh_time.strftime("%H:%M"),
                "reroll_nonce": reroll_nonce,
                "prompt_version": PROMPT_VERSION,
            }
        )

        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]

        background = None
        meta = self._read_json(meta_path)
        if os.path.exists(bg_path) and meta and meta.get("fingerprint") == fingerprint:
            try:
                with Image.open(bg_path) as img:
                    background = img.convert("RGB")
            except Exception:
                logger.exception("Failed to load cached background; regenerating.")
                background = None

        weather = None
        try:
            weather = self._fetch_weather_snapshot(owm_key, units, lat, lon, now)
        except Exception:
            logger.exception("Failed to fetch weather; continuing without overlay.")

        if background is None:
            prompt = self._build_prompt(weather, reroll_nonce=reroll_nonce, cache_id=cache_id, day_key=day_key)
            background = self._generate_gemini_background(
                api_key=gemini_key,
                prompt=prompt,
                model=model,
                image_size=image_size,
                aspect_ratio="9:16" if device_config.get_config("orientation") == "vertical" else "16:9",
            )
            background = resize_image(background, dimensions)
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
                },
            )
            try:
                background.save(latest_link)
            except Exception:
                logger.exception("Failed to update latest background pointer.")

        if weather:
            overlay = self._render_weather_overlay(weather, tz, forecast_days, dimensions)
            return self._composite(background, overlay)

        return background

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

    def _build_prompt(self, weather, reroll_nonce=0, cache_id="default", day_key=""):
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

        pad = max(10, int(width * 0.02))
        badge_h = max(72, int(height * 0.16))
        badge_w = max(250, int(width * 0.34))
        bar_h = max(110, int(height * 0.24))

        badge_box = (pad, pad, pad + badge_w, pad + badge_h)
        bar_box = (pad, height - pad - bar_h, width - pad, height - pad)

        self._rounded(draw, badge_box, fill=(255, 255, 255, 235), radius=16)
        self._rounded(draw, bar_box, fill=(255, 255, 255, 235), radius=18)

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

        title_font = self._font("Jost", int(height * 0.16))
        day_font = self._font("Jost", int(height * 0.18), bold=True)
        temp_font = self._font("Jost", int(height * 0.17))
        pop_font = self._font("Jost", int(height * 0.15))

        draw.text((x0 + pad, y0 + int(pad * 0.3)), f"Next {forecast_days} days", fill=(0, 0, 0, 255), font=title_font)

        daily = weather.daily[1 : 1 + forecast_days] if weather.daily else []
        cols = max(1, forecast_days)
        col_w = int((width - pad * 2) / cols)
        base_y = y0 + int(height * 0.28)

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
