"""Графики как inline-SVG, без единой внешней библиотеки.

Регламент прямо требует: «нет скрытых зависимостей от внешних ресурсов, не
поставляемых с решением». Подключить график с CDN — значит нарушить это и
получить пустую страницу у проверяющего без интернета. Поэтому рисуем сами:
двести строк на две фигуры дешевле, чем вендорить библиотеку графиков.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

WIDTH = 760
HEIGHT = 260
PADDING_LEFT = 64
PADDING_RIGHT = 16
PADDING_TOP = 16
PADDING_BOTTOM = 34

#: Цвета берутся из переменных темы страницы, чтобы график жил и в тёмной.
SERIES_COLORS = ("var(--accent)", "var(--accent-2)", "var(--accent-3)")


@dataclass(frozen=True)
class Series:
    name: str
    values: list[float]


def line_chart(labels: list[str], series: list[Series], *, unit: str = "") -> str:
    """Линейный график по общей оси X (обычно годы или месяцы)."""
    if not labels or not series or not any(item.values for item in series):
        return _empty("нет данных")

    high = max(max(item.values) for item in series)
    low = min(min(item.values) for item in series)
    low = min(low, 0.0)
    if high == low:
        high = low + 1.0

    plot_width = WIDTH - PADDING_LEFT - PADDING_RIGHT
    plot_height = HEIGHT - PADDING_TOP - PADDING_BOTTOM
    step = plot_width / max(len(labels) - 1, 1)

    def y_of(value: float) -> float:
        return PADDING_TOP + plot_height - (value - low) / (high - low) * plot_height

    parts = [_open(), *_grid(low, high, y_of, unit)]

    for index, item in enumerate(series):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        points = " ".join(
            f"{PADDING_LEFT + position * step:.1f},{y_of(value):.1f}"
            for position, value in enumerate(item.values)
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" points="{points}"/>'
        )

    parts.extend(_x_labels(labels, step))
    parts.append(_legend(series))
    parts.append("</svg>")
    return "\n".join(parts)


def bar_chart(labels: list[str], values: list[float], *, unit: str = "") -> str:
    """Столбцы со знаком: отрицательные значения рисуются вниз от нуля."""
    if not labels or not values:
        return _empty("нет данных")

    high = max(max(values), 0.0)
    low = min(min(values), 0.0)
    if high == low:
        high = low + 1.0

    plot_width = WIDTH - PADDING_LEFT - PADDING_RIGHT
    plot_height = HEIGHT - PADDING_TOP - PADDING_BOTTOM
    slot = plot_width / len(values)
    bar_width = max(slot * 0.65, 1.0)

    def y_of(value: float) -> float:
        return PADDING_TOP + plot_height - (value - low) / (high - low) * plot_height

    zero = y_of(0.0)
    parts = [_open(), *_grid(low, high, y_of, unit)]

    for index, value in enumerate(values):
        x = PADDING_LEFT + index * slot + (slot - bar_width) / 2
        y = min(y_of(value), zero)
        height = abs(y_of(value) - zero)
        color = "var(--accent)" if value >= 0 else "var(--danger)"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{max(height, 0.5):.1f}" fill="{color}" rx="2"/>'
        )

    parts.extend(_x_labels(labels, slot, offset=slot / 2))
    parts.append("</svg>")
    return "\n".join(parts)


def _open() -> str:
    return (
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="100%" height="{HEIGHT}" '
        f'role="img" class="chart" preserveAspectRatio="xMidYMid meet">'
    )


def _grid(low: float, high: float, y_of, unit: str) -> list[str]:
    parts = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = low + (high - low) * fraction
        y = y_of(value)
        parts.append(
            f'<line x1="{PADDING_LEFT}" y1="{y:.1f}" x2="{WIDTH - PADDING_RIGHT}" '
            f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PADDING_LEFT - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{_format(value)}{html.escape(unit)}</text>'
        )
    return parts


def _x_labels(labels: list[str], step: float, offset: float = 0.0) -> list[str]:
    every = max(1, len(labels) // 12)
    parts = []
    for index, label in enumerate(labels):
        if index % every:
            continue
        x = PADDING_LEFT + index * step + offset
        parts.append(
            f'<text x="{x:.1f}" y="{HEIGHT - 12}" text-anchor="middle" '
            f'class="tick">{html.escape(str(label))}</text>'
        )
    return parts


def _legend(series: list[Series]) -> str:
    parts = []
    for index, item in enumerate(series):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        x = PADDING_LEFT + index * 170
        parts.append(
            f'<rect x="{x}" y="{PADDING_TOP - 8}" width="10" height="10" fill="{color}" rx="2"/>'
            f'<text x="{x + 16}" y="{PADDING_TOP + 1}" class="tick">{html.escape(item.name)}</text>'
        )
    return "".join(parts)


def _empty(message: str) -> str:
    return (
        f'<svg viewBox="0 0 {WIDTH} 80" width="100%" height="80" class="chart">'
        f'<text x="{WIDTH / 2}" y="45" text-anchor="middle" class="tick">'
        f"{html.escape(message)}</text></svg>"
    )


def _format(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}k"
    if magnitude >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"
