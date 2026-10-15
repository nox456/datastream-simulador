"""Gráficos en tiempo real dibujados directamente con pygame."""

from __future__ import annotations

from typing import Callable, Sequence

import pygame

from .tema import BORDE, REJILLA, TEXTO_SUAVE, Fuentes

Serie = tuple[str, tuple, Sequence[float]]  # (nombre, color, valores)


def _mapear(valor: float, minimo: float, maximo: float, a: float, b: float) -> float:
    if maximo == minimo:
        return (a + b) / 2
    return a + (valor - minimo) * (b - a) / (maximo - minimo)


def grafico_lineas(
    superficie: pygame.Surface,
    fuentes: Fuentes,
    area: pygame.Rect,
    tiempos: Sequence[float],
    series: Sequence[Serie],
    y_min: float,
    y_max: float,
    formato_y: Callable[[float], str] = lambda v: f"{v:.0f}",
    titulo: str = "",
    leyenda: bool = True,
    x_desde: float | None = None,
    x_etiqueta: Callable[[float], str] = lambda t: f"{t:.0f} s",
) -> None:
    """Dibuja series sobre un eje X común (tiempo o iteración)."""
    zona = pygame.Rect(area.x + 44, area.y + (16 if titulo else 4), area.width - 52, area.height - (36 if titulo else 24))
    if titulo:
        fuentes.texto(superficie, titulo, (area.x + 2, area.y), 15, TEXTO_SUAVE)

    # Rejilla horizontal con etiquetas del eje Y
    for i in range(5):
        valor = y_min + (y_max - y_min) * i / 4
        y = _mapear(valor, y_min, y_max, zona.bottom, zona.top)
        pygame.draw.line(superficie, REJILLA, (zona.left, y), (zona.right, y))
        fuentes.texto(superficie, formato_y(valor), (zona.left - 5, int(y) - 6), 14, TEXTO_SUAVE, alineacion="derecha")
    pygame.draw.rect(superficie, BORDE, zona, width=1)

    if len(tiempos) < 2:
        fuentes.texto(superficie, "esperando datos...", zona.center, 15, TEXTO_SUAVE, alineacion="centro")
        return
    t_min = tiempos[0] if x_desde is None else min(x_desde, tiempos[0])
    t_max = max(tiempos[-1], t_min + 1e-6)
    fuentes.texto(superficie, x_etiqueta(t_min), (zona.left, zona.bottom + 3), 14, TEXTO_SUAVE)
    fuentes.texto(superficie, x_etiqueta(t_max), (zona.right, zona.bottom + 3), 14, TEXTO_SUAVE, alineacion="derecha")

    paso = max(1, len(tiempos) // zona.width)  # no dibujar más puntos que píxeles
    for _, color, valores in series:
        puntos = [
            (
                _mapear(t, t_min, t_max, zona.left, zona.right),
                _mapear(min(max(v, y_min), y_max), y_min, y_max, zona.bottom, zona.top),
            )
            for t, v in list(zip(tiempos, valores))[::paso]
        ]
        if len(puntos) >= 2:
            pygame.draw.lines(superficie, color, False, puntos, 2)

    if leyenda and len(series) > 1:
        x = zona.left + 6
        for nombre, color, _ in series:
            pygame.draw.rect(superficie, color, (x, zona.top + 6, 10, 10))
            rect = fuentes.texto(superficie, nombre, (x + 13, zona.top + 4), 14, TEXTO_SUAVE)
            x = rect.right + 10


def grafico_barras(
    superficie: pygame.Surface,
    fuentes: Fuentes,
    area: pygame.Rect,
    etiquetas: Sequence[str],
    valores: Sequence[float],
    colores: Sequence[tuple],
    valor_maximo: float,
    formato: Callable[[float], str] = lambda v: f"{v:.1f}",
    subetiquetas: Sequence[str] | None = None,
) -> None:
    n = len(valores)
    if n == 0:
        return
    base = area.bottom - (34 if subetiquetas else 18)
    alto_util = base - area.top - 18
    ancho = area.width / n
    pygame.draw.line(superficie, BORDE, (area.left, base), (area.right, base))
    for i, (etiqueta, valor, color) in enumerate(zip(etiquetas, valores, colores)):
        alto = alto_util * min(valor / valor_maximo, 1.0) if valor_maximo > 0 else 0
        barra = pygame.Rect(area.left + i * ancho + ancho * 0.2, base - alto, ancho * 0.6, alto)
        pygame.draw.rect(superficie, color, barra, border_radius=3)
        fuentes.texto(superficie, formato(valor), (barra.centerx, barra.top - 15), 15, color, alineacion="centro")
        fuentes.texto(superficie, etiqueta, (barra.centerx, base + 3), 15, TEXTO_SUAVE, alineacion="centro")
        if subetiquetas:
            fuentes.texto(superficie, subetiquetas[i], (barra.centerx, base + 18), 13, TEXTO_SUAVE, alineacion="centro")
