from __future__ import annotations

from PIL import ImageDraw


def _wrap(draw, text, font, max_width):
    words = (text or "").split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        cand = f"{current} {word}"
        if draw.textlength(cand, font=font) <= max_width:  # type: ignore[attr-defined]
            current = cand
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_family_banner(
    image,
    *,
    headline: str,
    detail: str = "",
    font_fn=None,
    region=None,
):
    if not headline:
        return image

    x0, y0, x1, y1 = region or (0, 0, image.size[0], image.size[1])
    if x1 <= x0 or y1 <= y0:
        return image

    draw = ImageDraw.Draw(image)
    w = x1 - x0
    h = y1 - y0

    margin = max(10, int(min(w, h) * 0.04))
    pad_x = max(10, int(margin * 0.9))
    pad_y = max(8, int(margin * 0.55))

    if font_fn:
        headline_font = font_fn("Jost", max(14, int(h * 0.055)), bold=True)
        detail_font = font_fn("Jost", max(12, int(h * 0.040)), bold=False)
    else:
        from PIL import ImageFont

        headline_font = ImageFont.load_default()
        detail_font = ImageFont.load_default()

    max_text_w = w - 2 * (margin + pad_x)

    headline_lines = _wrap(draw, headline, headline_font, max_text_w)
    detail_lines = _wrap(draw, detail, detail_font, max_text_w) if detail else []

    def line_h(font_obj):
        try:
            ascent, descent = font_obj.getmetrics()
            return int(ascent + descent)
        except Exception:
            bbox = draw.textbbox((0, 0), "Ag", font=font_obj)
            return bbox[3] - bbox[1]

    hh = line_h(headline_font)
    dh = line_h(detail_font)
    gap = max(4, int(h * 0.012))

    box_h = pad_y * 2 + (len(headline_lines) * hh)
    if detail_lines:
        box_h += gap + (len(detail_lines) * dh)

    box_w = w - 2 * margin
    bx0 = x0 + margin
    by0 = y0 + margin
    bx1 = bx0 + box_w
    by1 = min(y1 - margin, by0 + box_h)

    radius = max(8, int(min(box_w, box_h) * 0.06))
    try:
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=radius, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    except Exception:
        draw.rectangle((bx0, by0, bx1, by1), fill=(255, 255, 255))

    tx = bx0 + pad_x
    ty = by0 + pad_y
    for line in headline_lines:
        draw.text((tx, ty), line, fill=(0, 0, 0), font=headline_font)
        ty += hh
    if detail_lines:
        ty += gap
        for line in detail_lines:
            draw.text((tx, ty), line, fill=(60, 60, 60), font=detail_font)
            ty += dh

    return image

