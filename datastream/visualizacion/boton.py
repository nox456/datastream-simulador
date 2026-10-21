"""Botones clicables del dashboard (alternativa al teclado)."""

from __future__ import annotations

from typing import Callable

import pygame

from . import tema


class Boton:
    """Botón rectangular con texto, resaltado al pasar el mouse y una acción al hacer clic."""

    def __init__(self, rect: pygame.Rect, etiqueta: str | Callable[[], str], accion: Callable[[], None]):
        self.rect = rect
        self._etiqueta = etiqueta  # texto fijo o función que lo calcula (p. ej. "Pausa" / "Reanudar")
        self.accion = accion

    @property
    def etiqueta(self) -> str:
        return self._etiqueta() if callable(self._etiqueta) else self._etiqueta

    def contiene(self, punto: tuple[float, float]) -> bool:
        return self.rect.collidepoint(punto)

    def pulsar(self) -> None:
        self.accion()

    def dibujar(self, superficie: pygame.Surface, fuentes: tema.Fuentes, resaltado: bool = False) -> None:
        fondo = tema.mezclar(tema.PANEL, tema.AZUL, 0.35) if resaltado else (30, 36, 44)
        pygame.draw.rect(superficie, fondo, self.rect, border_radius=5)
        pygame.draw.rect(superficie, tema.AZUL if resaltado else tema.BORDE, self.rect, 1, border_radius=5)
        fuentes.texto(
            superficie, self.etiqueta, (self.rect.centerx, self.rect.y + 5), 15,
            tema.TEXTO if resaltado else tema.TEXTO_SUAVE, alineacion="centro",
        )


class BarraBotones:
    """Fila de botones con ancho ajustado a su texto; reparte el espacio disponible entre ellos."""

    def __init__(self, area: pygame.Rect, separacion: int = 6):
        self.area = area
        self.separacion = separacion
        self.botones: list[Boton] = []

    def agregar(self, etiqueta: str | Callable[[], str], accion: Callable[[], None]) -> Boton:
        boton = Boton(pygame.Rect(0, 0, 0, 0), etiqueta, accion)
        self.botones.append(boton)
        return boton

    def distribuir(self, fuentes: tema.Fuentes) -> None:
        """Cada botón recibe el ancho de su texto más una parte igual del espacio sobrante."""
        fuente = fuentes.obtener(15)
        anchos = [fuente.size(b.etiqueta)[0] + 16 for b in self.botones]
        libre = self.area.width - sum(anchos) - self.separacion * (len(self.botones) - 1)
        if libre < 0:
            raise ValueError("Los botones no caben en el área asignada.")
        extra = libre / len(self.botones)
        x = float(self.area.x)
        for boton, ancho in zip(self.botones, anchos):
            boton.rect = pygame.Rect(round(x), self.area.y, round(ancho + extra), self.area.height)
            x += ancho + extra + self.separacion

    def boton_en(self, punto: tuple[float, float]) -> Boton | None:
        return next((b for b in self.botones if b.contiene(punto)), None)

    def dibujar(self, superficie: pygame.Surface, fuentes: tema.Fuentes, punto_mouse: tuple[float, float] | None) -> None:
        for boton in self.botones:
            boton.dibujar(superficie, fuentes, resaltado=punto_mouse is not None and boton.contiene(punto_mouse))
