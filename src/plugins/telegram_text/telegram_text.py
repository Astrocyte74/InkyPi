import os
import logging
from typing import List

from PIL import Image, ImageDraw, ImageFont

from plugins.base_plugin.base_plugin import BasePlugin
from utils.app_utils import resolve_path
from utils.image_utils import resize_image

logger = logging.getLogger(__name__)


class TelegramText(BasePlugin):
    """Render short Telegram text messages with optional styling and background."""

    FONT_PATH = resolve_path("static/fonts/Jost-SemiBold.ttf")
    DEFAULT_BG = "#141414"
    TEXT_COLOR = "#FFFFFF"
    SHADOW_COLOR = "#000000"

    STYLES = {"simple", "caption", "sticky"}

    def generate_image(self, settings, device_config):
        text = (settings.get("text") or "").strip()
        if not text:
            raise RuntimeError("Message text is required.")

        style = (settings.get("style") or "caption").lower()
        if style not in self.STYLES:
            style = "caption"

        background_path = settings.get("background_path")
        background_color = settings.get("background_color") or self.DEFAULT_BG
        placement = (settings.get("placement") or "center").lower()
        if placement not in {"center", "bottom"}:
            placement = "center"

        width, height = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            width, height = height, width

        base_image = self._prepare_background(width, height, background_path, background_color)

        font_size = self._initial_font_size(width, height)
        font = ImageFont.truetype(self.FONT_PATH, font_size)
        if placement == "bottom":
            max_text_width = int(width * 0.90)
            # Reserve a bottom band for text; adapt font to fit this area
            area_ratio = float(settings.get("text_area_ratio") or 0.35)
            area_ratio = min(max(area_ratio, 0.2), 0.6)  # clamp for sanity
            max_text_height = int(height * area_ratio)
        else:
            max_text_width = int(width * 0.78)
            max_text_height = int(height * 0.78)
        text_lines, font = self._wrap_text(base_image, text, font, max_text_width, max_text_height)

        rendered = self._draw_text(base_image, text_lines, font, style, placement)
        return rendered.convert("RGB")

    def _prepare_background(self, width, height, background_path, background_color):
        image = None
        if background_path:
            try:
                if os.path.exists(background_path):
                    with Image.open(background_path) as bg:
                        image = resize_image(bg.convert("RGB"), (width, height))
                else:
                    logger.warning("Background path %s not found; falling back to solid colour.", background_path)
            except Exception:
                logger.exception("Failed to load background image %s", background_path)

        if image is None:
            image = Image.new("RGB", (width, height), background_color)

        return image

    def _initial_font_size(self, width, height):
        # Rough heuristic: choose font size relative to image height.
        return max(28, int(min(width, height) * 0.08))

    def _wrap_text(self, base_image, text, font, max_width, max_height):
        draw = ImageDraw.Draw(base_image)
        font_size = font.size
        words = text.split()

        while font_size >= 24:
            lines = self._wrap_words(words, draw, font, max_width)
            bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=self._line_spacing(font))
            text_height = bbox[3] - bbox[1]
            if text_height <= max_height:
                return lines, font
            font_size -= 4
            font = ImageFont.truetype(self.FONT_PATH, font_size)

        # Minimum font size fallback
        return self._wrap_words(words, draw, font, max_width), font

    def _wrap_words(self, words: List[str], draw: ImageDraw.ImageDraw, font, max_width: int) -> List[str]:
        lines: List[str] = []
        if not words:
            return [""]

        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _line_spacing(self, font):
        return int(font.size * 0.35)

    def _draw_text(self, base_image, lines, font, style, placement):
        width, height = base_image.size
        overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        line_spacing = self._line_spacing(font)
        text = "\n".join(lines)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding_x = font.size * 0.7
        padding_y = font.size * 0.8

        x = (width - text_width) / 2
        if placement == "bottom":
            margin_bottom = int(font.size * 0.6)
            y = height - (text_height + padding_y) - margin_bottom
            if y < padding_y:
                y = padding_y
        else:
            y = (height - text_height) / 2

        if style == "caption":
            box_coords = (
                x - padding_x,
                y - padding_y,
                x + text_width + padding_x,
                y + text_height + padding_y,
            )
            draw.rounded_rectangle(
                box_coords,
                radius=int(font.size * 0.5),
                fill=(0, 0, 0, 180),
            )
        elif style == "sticky":
            box_coords = (
                x - padding_x,
                y - padding_y,
                x + text_width + padding_x,
                y + text_height + padding_y,
            )
            draw.rounded_rectangle(
                box_coords,
                radius=int(font.size * 0.4),
                fill=(247, 226, 122, 230),
            )
            pin_radius = max(6, int(font.size * 0.3))
            pin_center = (int(box_coords[0] + pin_radius * 1.5), int(box_coords[1] + pin_radius * 1.5))
            draw.ellipse(
                (
                    pin_center[0] - pin_radius,
                    pin_center[1] - pin_radius,
                    pin_center[0] + pin_radius,
                    pin_center[1] + pin_radius,
                ),
                fill=(200, 54, 54, 255),
            )

        shadow_offset = int(font.size * 0.06)
        if style in {"simple", "caption"}:
            draw.multiline_text(
                (x + shadow_offset, y + shadow_offset),
                text,
                font=font,
                fill=self._shadow_rgba(shadow_offset),
                spacing=line_spacing,
                align="center",
            )

        fill_color = self.TEXT_COLOR if style != "sticky" else "#1A1A1A"
        draw.multiline_text(
            (x, y),
            text,
            font=font,
            fill=fill_color,
            spacing=line_spacing,
            align="center",
        )

        composed = Image.alpha_composite(base_image.convert("RGBA"), overlay)
        return composed

    def _shadow_rgba(self, offset):
        opacity = 140 if offset >= 1 else 110
        return (0, 0, 0, opacity)
