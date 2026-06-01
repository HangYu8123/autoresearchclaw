"""Small standard-library PNG chart helpers.

These helpers intentionally avoid matplotlib/Pillow so ResearchClaw can still
produce basic run figures in minimal virtual environments.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable

_COLORS = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _canvas(width: int, height: int) -> list[bytearray]:
    return [bytearray([255, 255, 255]) * width for _ in range(height)]


def _set_px(img: list[bytearray], x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= y < len(img) and 0 <= x < len(img[0]) // 3:
        offset = x * 3
        img[y][offset : offset + 3] = bytes(color)


def _rect(
    img: list[bytearray],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    width = len(img[0]) // 3
    height = len(img)
    xa, xb = sorted((max(0, x0), min(width - 1, x1)))
    ya, yb = sorted((max(0, y0), min(height - 1, y1)))
    row = bytes(color) * (xb - xa + 1)
    for y in range(ya, yb + 1):
        img[y][xa * 3 : (xb + 1) * 3] = row


def _line(
    img: list[bytearray],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    *,
    width: int = 1,
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for ox in range(-(width // 2), width // 2 + 1):
            for oy in range(-(width // 2), width // 2 + 1):
                _set_px(img, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _write_png(path: Path, img: list[bytearray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = len(img[0]) // 3
    height = len(img)
    raw = b"".join(b"\x00" + bytes(row) for row in img)

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(0.0, min(values))
    hi = max(0.0, max(values))
    if hi == lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _scale(value: float, lo: float, hi: float, top: int, bottom: int) -> int:
    frac = (value - lo) / (hi - lo) if hi != lo else 0.5
    return int(bottom - frac * (bottom - top))


def save_bar_chart(
    output_path: str | Path,
    labels: list[str],
    values: list[float],
    *,
    ci_low: list[float] | None = None,
    ci_high: list[float] | None = None,
    width: int = 960,
    height: int = 540,
) -> Path | None:
    """Write a simple vertical bar chart PNG."""
    values = _finite(values)
    if not labels or not values:
        return None
    labels = labels[: len(values)]
    img = _canvas(width, height)
    axis = (45, 45, 45)
    grid = (225, 225, 225)
    left, right, top, bottom = 84, width - 36, 44, height - 76
    lo, hi = _bounds(values + _finite(ci_low or []) + _finite(ci_high or []))
    zero_y = _scale(0.0, lo, hi, top, bottom)
    for i in range(6):
        y = top + int((bottom - top) * i / 5)
        _line(img, left, y, right, y, grid)
    _line(img, left, top, left, bottom, axis, width=2)
    _line(img, left, zero_y, right, zero_y, axis, width=2)

    n = len(values)
    slot = max(1, (right - left) / n)
    bar_w = max(8, int(slot * 0.58))
    for i, value in enumerate(values):
        cx = int(left + slot * (i + 0.5))
        y = _scale(value, lo, hi, top, bottom)
        color = _rgb(_COLORS[i % len(_COLORS)])
        _rect(img, cx - bar_w // 2, min(y, zero_y), cx + bar_w // 2, max(y, zero_y), color)
        if ci_low and ci_high and i < len(ci_low) and i < len(ci_high):
            low_y = _scale(float(ci_low[i]), lo, hi, top, bottom)
            high_y = _scale(float(ci_high[i]), lo, hi, top, bottom)
            _line(img, cx, high_y, cx, low_y, axis, width=2)
            _line(img, cx - 8, high_y, cx + 8, high_y, axis)
            _line(img, cx - 8, low_y, cx + 8, low_y, axis)
    return _write_png(Path(output_path), img)


def save_grouped_bar_chart(
    output_path: str | Path,
    labels: list[str],
    metric_names: list[str],
    data_matrix: list[list[float]],
    *,
    width: int = 1040,
    height: int = 560,
) -> Path | None:
    values = _finite(v for row in data_matrix for v in row)
    if not labels or not metric_names or not values:
        return None
    img = _canvas(width, height)
    axis = (45, 45, 45)
    left, right, top, bottom = 84, width - 36, 44, height - 76
    lo, hi = _bounds(values)
    zero_y = _scale(0.0, lo, hi, top, bottom)
    _line(img, left, top, left, bottom, axis, width=2)
    _line(img, left, zero_y, right, zero_y, axis, width=2)
    slot = max(1, (right - left) / len(labels))
    group_w = int(slot * 0.74)
    bar_w = max(4, group_w // max(1, len(metric_names)))
    for i, row in enumerate(data_matrix[: len(labels)]):
        start = int(left + slot * i + (slot - group_w) / 2)
        for j, raw in enumerate(row[: len(metric_names)]):
            value = float(raw)
            x0 = start + j * bar_w
            x1 = x0 + max(3, bar_w - 2)
            y = _scale(value, lo, hi, top, bottom)
            _rect(img, x0, min(y, zero_y), x1, max(y, zero_y), _rgb(_COLORS[j % len(_COLORS)]))
    return _write_png(Path(output_path), img)


def save_heatmap_chart(
    output_path: str | Path,
    row_labels: list[str],
    col_labels: list[str],
    data_matrix: list[list[float]],
    *,
    width: int = 960,
    height: int = 540,
) -> Path | None:
    values = _finite(v for row in data_matrix for v in row)
    if not row_labels or not col_labels or not values:
        return None
    img = _canvas(width, height)
    left, right, top, bottom = 96, width - 44, 48, height - 64
    rows, cols = len(row_labels), len(col_labels)
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1.0
    cell_w = max(1, (right - left) // cols)
    cell_h = max(1, (bottom - top) // rows)
    for i, row in enumerate(data_matrix[:rows]):
        for j, raw in enumerate(row[:cols]):
            frac = (float(raw) - lo) / (hi - lo)
            color = (
                int(245 - frac * 175),
                int(248 - frac * 122),
                int(255 - frac * 55),
            )
            _rect(
                img,
                left + j * cell_w,
                top + i * cell_h,
                left + (j + 1) * cell_w - 2,
                top + (i + 1) * cell_h - 2,
                color,
            )
    _line(img, left, top, left, bottom, (45, 45, 45), width=2)
    _line(img, left, bottom, right, bottom, (45, 45, 45), width=2)
    return _write_png(Path(output_path), img)


def save_line_chart(
    output_path: str | Path,
    series_data: list[dict[str, Any]],
    *,
    width: int = 960,
    height: int = 540,
) -> Path | None:
    points: list[tuple[int, float]] = []
    for series in series_data:
        ys = _finite(series.get("values") or series.get("y") or [])
        points.extend((idx, value) for idx, value in enumerate(ys))
    if not points:
        return None
    img = _canvas(width, height)
    left, right, top, bottom = 84, width - 36, 44, height - 76
    lo, hi = _bounds([v for _, v in points])
    max_x = max(x for x, _ in points) or 1
    axis = (45, 45, 45)
    _line(img, left, top, left, bottom, axis, width=2)
    _line(img, left, bottom, right, bottom, axis, width=2)
    for si, series in enumerate(series_data):
        ys = _finite(series.get("values") or series.get("y") or [])
        if not ys:
            continue
        prev: tuple[int, int] | None = None
        color = _rgb(_COLORS[si % len(_COLORS)])
        for idx, value in enumerate(ys):
            x = int(left + (right - left) * idx / max_x)
            y = _scale(value, lo, hi, top, bottom)
            _rect(img, x - 3, y - 3, x + 3, y + 3, color)
            if prev:
                _line(img, prev[0], prev[1], x, y, color, width=2)
            prev = (x, y)
    return _write_png(Path(output_path), img)


def save_scatter_chart(
    output_path: str | Path,
    groups: list[dict[str, Any]],
    *,
    width: int = 960,
    height: int = 540,
) -> Path | None:
    xs = _finite(x for g in groups for x in (g.get("x") or []))
    ys = _finite(y for g in groups for y in (g.get("y") or []))
    if not xs or not ys:
        return None
    img = _canvas(width, height)
    left, right, top, bottom = 84, width - 36, 44, height - 76
    xlo, xhi = _bounds(xs)
    ylo, yhi = _bounds(ys)
    axis = (45, 45, 45)
    _line(img, left, top, left, bottom, axis, width=2)
    _line(img, left, bottom, right, bottom, axis, width=2)
    for gi, group in enumerate(groups):
        color = _rgb(_COLORS[gi % len(_COLORS)])
        for xraw, yraw in zip(group.get("x") or [], group.get("y") or []):
            xval = float(xraw)
            yval = float(yraw)
            x = int(left + (xval - xlo) / (xhi - xlo) * (right - left))
            y = _scale(yval, ylo, yhi, top, bottom)
            _rect(img, x - 4, y - 4, x + 4, y + 4, color)
    return _write_png(Path(output_path), img)
