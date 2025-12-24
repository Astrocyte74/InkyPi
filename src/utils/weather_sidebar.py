from __future__ import annotations

from datetime import datetime, timezone

from PIL import Image, ImageDraw


def render_weather_sidebar_panel(
    weather,
    tz,
    forecast_days,
    sidebar_size,
    location_label="",
    *,
    font,
    icon_renderer,
):
    panel_w, panel_h = sidebar_size
    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    def _text_bbox(text, font_obj):
        try:
            return draw.textbbox((0, 0), text, font=font_obj)
        except Exception:
            try:
                box = font_obj.getbbox(text)
                return (0, 0, box[2] - box[0], box[3] - box[1])
            except Exception:
                w, h = font_obj.getsize(text)  # type: ignore[attr-defined]
                return (0, 0, w, h)

    def _text_size(text, font_obj):
        b = _text_bbox(text, font_obj)
        return (b[2] - b[0], b[3] - b[1])

    def _wrap_text(text, font_obj, max_width, max_lines=2):
        words = (text or "").strip().split()
        if not words:
            return []

        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            w, _ = _text_size(candidate, font_obj)
            if w <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)

        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]
            while True:
                w, _ = _text_size(lines[-1] + "…", font_obj)
                if w <= max_width or len(lines[-1]) <= 1:
                    break
                lines[-1] = lines[-1].rsplit(" ", 1)[0]
            lines[-1] = lines[-1] + "…"
        return lines

    def _line_height(font_obj):
        try:
            ascent, descent = font_obj.getmetrics()
            return max(1, int(ascent + descent))
        except Exception:
            return max(1, _text_size("Ag", font_obj)[1])

    draw.line((0, 0, 0, panel_h), fill=(0, 0, 0))
    pad = max(10, int(panel_w * 0.06))
    gap = max(6, int(pad * 0.5))

    title_font = font("Jost", max(14, int(panel_w * 0.085)), bold=True)
    base_temp_font_size = max(18, int(panel_w * 0.16))
    small_font = font("Jost", max(12, int(panel_w * 0.065)))
    location_font = font("Jost", max(12, int(panel_w * 0.070)))
    text_secondary = (0, 0, 0)
    icon_accent = (0, 0, 0, 255)
    hi_lo_base_size = max(14, int(panel_w * 0.10))

    y = pad
    if not weather:
        draw.text((pad, y), "Weather", fill=(0, 0, 0), font=title_font)
        draw.text((pad, y + int(pad * 1.4)), "Unavailable", fill=(0, 0, 0), font=small_font)
        return panel

    def _format_degree(value):
        try:
            value = round(float(value))
        except (TypeError, ValueError):
            value = 0
        if weather.units == "standard":
            return f"{value}K"
        return f"{value}°"

    if location_label:
        draw.text((pad, y), location_label, fill=text_secondary, font=location_font)
        y += _line_height(location_font) + int(gap * 0.8)

    temp_value = round(weather.current_temp)
    feels_value = round(weather.feels_like)
    temp_unit = "°C" if weather.units == "metric" else ("°F" if weather.units == "imperial" else "K")
    desc = (weather.description or "").strip().capitalize()

    icon_size = max(40, int(panel_w * 0.22))
    icon = icon_renderer(weather.icon, size=icon_size).convert("RGBA")

    icon_x = pad
    icon_y = y
    panel.paste(icon, (icon_x, icon_y), icon)

    text_x = icon_x + icon_size + gap
    max_text_w = panel_w - pad - text_x
    max_text_w = max(40, max_text_w)

    def _fit_font(text, max_width, start_size, min_size=12):
        size = start_size
        while size > min_size:
            font_obj = font("Jost", size, bold=True)
            if _text_size(text, font_obj)[0] <= max_width:
                return font_obj
            size -= 2
        return font("Jost", min_size, bold=True)

    small_lh = _line_height(small_font)
    line_gap = max(6, int(small_lh * 0.35))
    y_cursor = icon_y

    temp_font = _fit_font(f"{temp_value}{temp_unit}", max_text_w, base_temp_font_size, min_size=14)
    temp_lh = _line_height(temp_font)
    temp_y = icon_y + max(0, int((icon_size - temp_lh) / 2))
    draw.text((text_x, temp_y), f"{temp_value}{temp_unit}", fill=(0, 0, 0), font=temp_font)

    y_cursor = icon_y + max(icon_size, (temp_y - icon_y) + temp_lh) + int(gap * 0.7)

    feels_line = f"Feels like {feels_value}{temp_unit}"
    draw.text((text_x, y_cursor), feels_line, fill=text_secondary, font=small_font)
    y_cursor += small_lh + line_gap

    today_high_low = ""
    try:
        today = (weather.daily or [None])[0] or {}
        temps = today.get("temp") or {}
        high = temps.get("max")
        low = temps.get("min")
        if high is not None and low is not None:
            today_high_low = f"H: {_format_degree(high)}  L: {_format_degree(low)}"
    except Exception:
        today_high_low = ""

    if today_high_low:
        hi_lo_font = _fit_font(today_high_low, max_text_w, hi_lo_base_size, min_size=12)
        draw.text((text_x, y_cursor), today_high_low, fill=(0, 0, 0), font=hi_lo_font)
        y_cursor += _line_height(hi_lo_font) + line_gap

    header_h = max(icon_size, y_cursor - icon_y)
    divider_y = icon_y + header_h + int(pad * 0.7)
    draw.line((pad, divider_y, panel_w - pad, divider_y), fill=(0, 0, 0))
    y = divider_y + int(pad * 0.7)

    daily = weather.daily[1 : 1 + forecast_days] if weather.daily else []
    if not daily:
        draw.text((pad, y), "No forecast", fill=(0, 0, 0), font=small_font)
        return panel

    remaining_h = panel_h - y - pad
    min_row_h = 96 if forecast_days <= 3 else 72
    base_row_h = max(min_row_h, int(remaining_h / max(1, forecast_days)))

    def _layout_forecast(scale=1.0):
        row_pad = max(6, int(base_row_h * 0.12 * scale))
        row_line_gap = max(5, int(row_pad * 0.6))
        row_icon = max(28, int(base_row_h * 0.62 * scale))

        main_font_size = max(13, int(base_row_h * 0.25 * scale))
        precip_font_size = max(12, int(base_row_h * 0.22 * scale))

        row_main_font = font("Jost", main_font_size, bold=True)
        row_precip_font = font("Jost", precip_font_size)

        main_lh = _line_height(row_main_font)
        precip_lh = _line_height(row_precip_font)

        return {
            "row_pad": row_pad,
            "row_line_gap": row_line_gap,
            "row_icon": row_icon,
            "main_font_size": main_font_size,
            "row_main_font": row_main_font,
            "row_precip_font": row_precip_font,
            "main_lh": main_lh,
            "precip_lh": precip_lh,
        }

    layout = _layout_forecast(scale=1.0)
    for _ in range(3):
        inter_gap = max(6, int(layout["row_pad"] * 0.6))
        total_needed = 0
        for day in daily[:forecast_days]:
            total_text_h = layout["main_lh"] + layout["row_line_gap"] + layout["precip_lh"]
            content_h = max(layout["row_icon"], total_text_h)
            total_needed += content_h + layout["row_pad"] * 2
        total_needed += inter_gap * max(0, forecast_days - 1)

        if total_needed <= remaining_h or total_needed <= 0:
            break
        layout = _layout_forecast(scale=(remaining_h / total_needed) * 0.98)

    y_cursor = y
    inter_gap = max(6, int(layout["row_pad"] * 0.6))

    def _fit_bold(text, max_width, start_size, min_size=10):
        size = start_size
        while size > min_size:
            font_obj = font("Jost", size, bold=True)
            if _text_size(text, font_obj)[0] <= max_width:
                return font_obj
            size -= 1
        return font("Jost", min_size, bold=True)

    for idx, day in enumerate(daily[:forecast_days]):
        total_text_h = layout["main_lh"] + layout["row_line_gap"] + layout["precip_lh"]
        content_h = max(layout["row_icon"], total_text_h)
        block_h = content_h + layout["row_pad"] * 2

        row_y0 = y_cursor
        row_y1 = row_y0 + block_h
        if row_y0 >= panel_h - pad or row_y1 > panel_h - pad:
            break

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

        row_pad = layout["row_pad"]
        icon_img = icon_renderer(icon_code, size=layout["row_icon"]).convert("RGBA")
        icon_y = row_y0 + row_pad
        panel.paste(icon_img, (pad, icon_y), icon_img)

        text_x = pad + layout["row_icon"] + gap
        max_text_w = max(30, panel_w - pad - text_x)

        line_y = row_y0 + row_pad
        main_line = f"{label}  H: {_format_degree(high)}  L: {_format_degree(low)}"
        main_font = _fit_bold(main_line, max_text_w, layout["main_font_size"], min_size=10)
        draw.text((text_x, line_y), main_line, fill=(0, 0, 0), font=main_font)
        line_y += _line_height(main_font) + layout["row_line_gap"]

        precip_line = f"Precip: {pop}%"
        draw.text((text_x, line_y), precip_line, fill=(0, 0, 0), font=layout["row_precip_font"])

        if idx < forecast_days - 1:
            divider_y = row_y1 + int(inter_gap / 2)
            if divider_y < panel_h - pad:
                draw.line((pad, divider_y, panel_w - pad, divider_y), fill=(0, 0, 0))
        y_cursor = row_y1 + inter_gap

    return panel

