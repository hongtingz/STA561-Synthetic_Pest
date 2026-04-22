"""Preview rendering for layout-spec inspection."""

from __future__ import annotations

from pathlib import Path

from prob_ml.layout import LayoutSpec, summarize_layout_decisions


def _world_to_canvas(
    point: tuple[float, float],
    room_width: float,
    room_depth: float,
    origin_x: int,
    origin_y: int,
    scale: float,
) -> tuple[int, int]:
    x, y = point
    canvas_x = origin_x + int(round((x + room_width / 2.0) * scale))
    canvas_y = origin_y + int(round((room_depth / 2.0 - y) * scale))
    return canvas_x, canvas_y


def _rgb(color: tuple[float, float, float]) -> str:
    return f"rgb({int(color[0] * 255)},{int(color[1] * 255)},{int(color[2] * 255)})"


def save_layout_preview(layout_spec: LayoutSpec, output_path: Path) -> None:
    """Draw a 2D top-down SVG preview of the inferred layout."""
    diagnostics = summarize_layout_decisions(layout_spec)
    room = layout_spec.room
    image_width = 1120
    image_height = 840
    sidebar_width = 320
    margin = 48
    room_area_width = image_width - sidebar_width - margin * 2
    room_area_height = image_height - margin * 2
    scale = min(room_area_width / room.width, room_area_height / room.depth)
    origin_x = margin
    origin_y = margin

    room_x0 = origin_x
    room_y0 = origin_y
    room_x1 = room_x0 + int(round(room.width * scale))
    room_y1 = room_y0 + int(round(room.depth * scale))
    sidebar_x = room_x1 + 36

    elements: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{image_width}" '
            f'height="{image_height}" viewBox="0 0 {image_width} {image_height}">'
        ),
        '<rect width="100%" height="100%" fill="rgb(249,246,240)" />',
        (
            f'<rect x="{room_x0}" y="{room_y0}" '
            f'width="{room_x1 - room_x0}" height="{room_y1 - room_y0}" '
            'fill="rgb(241,236,228)" stroke="rgb(54,54,54)" stroke-width="3" />'
        ),
        (
            f'<rect x="{sidebar_x - 16}" y="{margin}" '
            f'width="{image_width - margin - (sidebar_x - 16)}" '
            f'height="{image_height - margin * 2}" fill="white" '
            'stroke="rgb(210,205,198)" stroke-width="2" />'
        ),
        (
            "<style>text { font-family: Menlo, Consolas, monospace; "
            "font-size: 14px; fill: rgb(20,20,20); }</style>"
        ),
    ]

    for fixture in layout_spec.fixtures:
        fx, fy, _ = fixture.center
        sx, sy, _ = fixture.size
        top_left = _world_to_canvas(
            (fx - sx / 2.0, fy + sy / 2.0),
            room.width,
            room.depth,
            origin_x,
            origin_y,
            scale,
        )
        bottom_right = _world_to_canvas(
            (fx + sx / 2.0, fy - sy / 2.0),
            room.width,
            room.depth,
            origin_x,
            origin_y,
            scale,
        )
        x = min(top_left[0], bottom_right[0])
        y = min(top_left[1], bottom_right[1])
        width = abs(bottom_right[0] - top_left[0])
        height = abs(bottom_right[1] - top_left[1])
        elements.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{_rgb(fixture.color)}" '
            'stroke="rgb(40,40,40)" stroke-width="2" />'
        )
        elements.append(f'<text x="{x + 6}" y="{y + 18}">{fixture.name}</text>')

    pest_colors = {
        "mouse": "rgb(201,90,74)",
        "rat": "rgb(126,94,69)",
        "cockroach": "rgb(120,55,18)",
    }
    for pest in layout_spec.pests:
        start = _world_to_canvas(
            (pest.path.start[0], pest.path.start[1]),
            room.width,
            room.depth,
            origin_x,
            origin_y,
            scale,
        )
        end = _world_to_canvas(
            (pest.path.end[0], pest.path.end[1]),
            room.width,
            room.depth,
            origin_x,
            origin_y,
            scale,
        )
        color = pest_colors.get(pest.pest_type, "rgb(180,40,40)")
        elements.append(
            f'<line x1="{start[0]}" y1="{start[1]}" '
            f'x2="{end[0]}" y2="{end[1]}" stroke="{color}" stroke-width="4" />'
        )
        elements.append(
            f'<circle cx="{start[0]}" cy="{start[1]}" '
            f'r="7" fill="{color}" stroke="black" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{start[0] + 10}" y="{start[1] - 8}" '
            f'fill="{color}">{pest.pest_id}</text>'
        )

    lines = [
        "Layout Preview",
        "",
        f"source: {Path(layout_spec.source_photo).name}",
        (
            f"room: {diagnostics['room']['width']} x "
            f"{diagnostics['room']['depth']} x {diagnostics['room']['height']}"
        ),
        f"camera lens: {diagnostics['camera']['lens_mm']} mm",
        f"fridge side: {diagnostics['fixture_summary']['fridge_side']}",
        f"fixtures: {diagnostics['fixture_summary']['fixture_count']}",
        f"pests: {', '.join(diagnostics['pest_types'])}",
        "",
        "photo cues",
        f"top brightness: {diagnostics['photo_cues']['brightness_top']}",
        f"mid brightness: {diagnostics['photo_cues']['brightness_mid']}",
        f"bottom brightness: {diagnostics['photo_cues']['brightness_bottom']}",
        f"left brightness: {diagnostics['photo_cues']['left_brightness']}",
        f"center brightness: {diagnostics['photo_cues']['center_brightness']}",
        f"right brightness: {diagnostics['photo_cues']['right_brightness']}",
        f"floor line ratio: {diagnostics['photo_cues']['floor_line_ratio']}",
        f"clutter score: {diagnostics['photo_cues']['clutter_score']}",
        f"warm bias: {diagnostics['photo_cues']['warm_bias']}",
        f"contrast score: {diagnostics['photo_cues']['contrast_score']}",
    ]

    cursor_y = margin + 18
    for line in lines:
        if line:
            safe = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            elements.append(f'<text x="{sidebar_x}" y="{cursor_y}">{safe}</text>')
            cursor_y += 22
        else:
            cursor_y += 12

    elements.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")
