"""Colores, fuentes y utilidades de dibujo compartidas por el dashboard."""

from __future__ import annotations

import pygame

FONDO = (13, 17, 23)
PANEL = (22, 27, 34)
BORDE = (48, 54, 61)
REJILLA = (36, 41, 48)
TEXTO = (230, 237, 243)
TEXTO_SUAVE = (139, 148, 158)
VERDE = (63, 185, 80)
AMARILLO = (210, 153, 34)
NARANJA = (240, 136, 62)
ROJO = (248, 81, 73)
AZUL = (88, 166, 255)
CIAN = (57, 208, 216)
INACTIVO = (72, 79, 88)

COLORES_CLUSTER = [AZUL, VERDE, (188, 140, 255), NARANJA, (255, 123, 114), CIAN]
COLORES_TAREA = [(121, 192, 255), (126, 231, 135), (255, 166, 87), (210, 168, 255), (255, 214, 102), (255, 123, 114)]
COLORES_EVENTO = {"falla": ROJO, "reparacion": VERDE, "ruta": AMARILLO, "emergencia": NARANJA, "info": TEXTO_SUAVE}


def mezclar(color_a: tuple, color_b: tuple, t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))


def color_escala(valor: float) -> tuple[int, int, int]:
    """0 → verde, 0.5 → amarillo, 1 → rojo."""
    if valor < 0.5:
        return mezclar(VERDE, AMARILLO, valor * 2)
    return mezclar(AMARILLO, ROJO, (valor - 0.5) * 2)


class Fuentes:
    """Caché de fuentes (la fuente por defecto de pygame incluye tildes y letras griegas)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, bool], pygame.font.Font] = {}

    def obtener(self, tamano: int, negrita: bool = False) -> pygame.font.Font:
        clave = (tamano, negrita)
        if clave not in self._cache:
            fuente = pygame.font.Font(None, tamano)
            fuente.set_bold(negrita)
            self._cache[clave] = fuente
        return self._cache[clave]

    def texto(
        self,
        superficie: pygame.Surface,
        contenido: str,
        posicion: tuple[int, int],
        tamano: int = 18,
        color: tuple = TEXTO,
        negrita: bool = False,
        alineacion: str = "izquierda",
    ) -> pygame.Rect:
        imagen = self.obtener(tamano, negrita).render(contenido, True, color)
        rect = imagen.get_rect()
        if alineacion == "centro":
            rect.midtop = posicion
        elif alineacion == "derecha":
            rect.topright = posicion
        else:
            rect.topleft = posicion
        superficie.blit(imagen, rect)
        return rect


def dibujar_panel(superficie: pygame.Surface, fuentes: Fuentes, rect: pygame.Rect, titulo: str) -> pygame.Rect:
    """Dibuja el fondo de un panel y devuelve el área útil bajo el título."""
    pygame.draw.rect(superficie, PANEL, rect, border_radius=8)
    pygame.draw.rect(superficie, BORDE, rect, width=1, border_radius=8)
    fuentes.texto(superficie, titulo, (rect.x + 12, rect.y + 9), 20, TEXTO, negrita=True)
    return pygame.Rect(rect.x + 10, rect.y + 32, rect.width - 20, rect.height - 42)


def linea_discontinua(superficie, color, inicio, fin, ancho=1, segmento=8) -> None:
    (x1, y1), (x2, y2) = inicio, fin
    largo = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1e-6)
    pasos = int(largo / segmento)
    for i in range(0, pasos, 2):
        a, b = i / pasos, min((i + 1) / pasos, 1)
        pygame.draw.line(
            superficie,
            color,
            (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a),
            (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b),
            ancho,
        )


def triangulo(superficie, color, centro, tamano=6, hacia_arriba=True) -> None:
    x, y = centro
    s = tamano if hacia_arriba else -tamano
    pygame.draw.polygon(superficie, color, [(x, y - s), (x - tamano, y + s), (x + tamano, y + s)])


def rombo(superficie, color, centro, tamano=6) -> None:
    x, y = centro
    pygame.draw.polygon(superficie, color, [(x, y - tamano), (x + tamano, y), (x, y + tamano), (x - tamano, y)])
