"""Mapa de la topología: tráfico en movimiento, rutas óptimas y fallas."""

from __future__ import annotations

import math

import pygame

from ..dominio.red import TipoNodo
from ..dominio.servidor import Servidor
from ..simulacion.centro_datos import CambioRuta, SimuladorCentroDatos
from .tema import (
    AMARILLO,
    AZUL,
    BORDE,
    CIAN,
    COLORES_CLUSTER,
    COLORES_TAREA,
    FONDO,
    INACTIVO,
    NARANJA,
    PANEL,
    ROJO,
    TEXTO,
    TEXTO_SUAVE,
    Fuentes,
    color_escala,
    dibujar_panel,
    linea_discontinua,
    mezclar,
)

COLOR_NODO = {
    TipoNodo.GATEWAY: (139, 92, 246),
    TipoNodo.NUCLEO: (31, 111, 235),
    TipoNodo.AGREGACION: (35, 134, 54),
    TipoNodo.ACCESO: (158, 106, 3),
}
DURACION_RESALTADO = 3.0  # segundos reales que se resalta un cambio de ruta
ANCHO_SERVIDOR, ALTO_SERVIDOR = 66, 50


def texto_ruta(ruta) -> str:
    return " > ".join(ruta) if ruta else "sin ruta"


class PanelTopologia:
    def __init__(self, simulador: SimuladorCentroDatos, rect: pygame.Rect, fuentes: Fuentes):
        self.sim = simulador
        self.rect = rect
        self.fuentes = fuentes
        self.cluster_seleccionado = 0
        self._cambio_resaltado: CambioRuta | None = None
        self._inicio_resaltado = 0.0
        self._cambios_vistos = 0
        self.indice_tipo = {t.nombre: i for i, t in enumerate(simulador.config.tipos_tarea)}
        self.posiciones = self._calcular_posiciones()

    # ------------------------------------------------------------------ #
    def _calcular_posiciones(self) -> dict[str, tuple[int, int]]:
        etapas = self.sim.topologia.etapas
        area = pygame.Rect(self.rect.x + 40, self.rect.y + 140, self.rect.width - 80, self.rect.height - 250)
        posiciones = {}
        for k, etapa in enumerate(etapas[:-1]):
            y = area.top + area.height * k / (len(etapas) - 1)
            for i, nodo in enumerate(etapa):
                posiciones[nodo.id] = (int(area.left + area.width * (i + 0.5) / len(etapa)), int(y))

        # Servidores agrupados bajo su clúster; el switch ToR queda centrado sobre ellos.
        ancho_cluster = area.width / len(self.sim.clusters)
        for c, cluster in enumerate(self.sim.clusters):
            inicio = area.left + c * ancho_cluster
            posiciones[cluster.switch] = (int(inicio + ancho_cluster / 2), posiciones[cluster.switch][1])
            for i, servidor in enumerate(cluster.servidores):
                x = inicio + ancho_cluster * (i + 0.5) / len(cluster.servidores)
                posiciones[servidor.id] = (int(x), area.bottom)
        return posiciones

    def seleccionar_cluster(self, indice: int) -> bool:
        if 0 <= indice < len(self.sim.clusters):
            self.cluster_seleccionado = indice
            return True
        return False

    def _marco_cluster(self, indice: int) -> pygame.Rect:
        """Recuadro del clúster en la parte inferior del panel (se usa para dibujar y para los clics)."""
        ancho = (self.rect.width - 80) / len(self.sim.clusters)
        x = self.rect.x + 40 + ancho * indice
        return pygame.Rect(x + 4, self.rect.bottom - 64, ancho - 8, 40)

    def cluster_en(self, punto: tuple[float, float]) -> int | None:
        """Índice del clúster cuyo recuadro contiene el punto, o None."""
        return next((i for i in range(len(self.sim.clusters)) if self._marco_cluster(i).collidepoint(punto)), None)

    def actualizar(self, tiempo_real: float) -> None:
        """Detecta cambios de ruta nuevos para resaltarlos."""
        cambios = self.sim.cambios_ruta
        if len(cambios) > self._cambios_vistos:
            self._cambio_resaltado = cambios[-1]
            self._inicio_resaltado = tiempo_real
            self._cambios_vistos = len(cambios)

    # ------------------------------------------------------------------ #
    def dibujar(self, superficie: pygame.Surface, tiempo_real: float) -> None:
        dibujar_panel(superficie, self.fuentes, self.rect, "Topología de la red · ruta más corta por programación dinámica")
        self._dibujar_enlaces(superficie)
        self._dibujar_ruta_seleccionada(superficie)
        self._dibujar_cambio_ruta(superficie, tiempo_real)
        self._dibujar_paquetes(superficie)
        self._dibujar_nodos(superficie, tiempo_real)
        self._dibujar_clusters(superficie)
        self._dibujar_leyenda(superficie)

    def _dibujar_enlaces(self, superficie: pygame.Surface) -> None:
        nodos = self.sim.topologia.nodos
        for (origen, destino), enlace in self.sim.topologia.enlaces.items():
            a, b = self.posiciones[origen], self.posiciones[destino]
            if not (nodos[origen].activo and nodos[destino].activo):
                linea_discontinua(superficie, INACTIVO, a, b, 1)
                continue
            nivel = min(enlace.congestion / 4, 1.0)  # 4 paquetes simultáneos = rojo
            pygame.draw.line(superficie, color_escala(nivel), a, b, 2 + min(enlace.en_transito, 4))

    def _polilinea(self, superficie, ruta, color, ancho) -> None:
        if ruta and len(ruta) > 1:
            pygame.draw.lines(superficie, color, False, [self.posiciones[n] for n in ruta], ancho)

    def _dibujar_ruta_seleccionada(self, superficie: pygame.Surface) -> None:
        indice = self.cluster_seleccionado
        cluster = self.sim.clusters[indice]
        ruta = self.sim.ruta_de_cluster(indice)
        self._polilinea(superficie, ruta, CIAN, 3)
        latencia = self.sim.latencias_actuales.get(cluster.nombre, float("inf"))
        detalle = f"{latencia:.2f} s" if ruta else "inalcanzable"
        self.fuentes.texto(
            superficie,
            f"[{indice + 1}] Ruta óptima a {cluster.nombre}: {texto_ruta(ruta)}  ·  f = {detalle}",
            (self.rect.x + 14, self.rect.y + 36),
            17,
            CIAN,
        )

    def _dibujar_cambio_ruta(self, superficie: pygame.Surface, tiempo_real: float) -> None:
        cambio = self._cambio_resaltado
        transcurrido = tiempo_real - self._inicio_resaltado
        if cambio is None or transcurrido > DURACION_RESALTADO:
            texto = "Sin cambios de ruta recientes" if cambio is None else (
                f"Último cambio (t = {cambio.t:.1f} s): {cambio.cluster} por {cambio.motivo}"
            )
            self.fuentes.texto(superficie, texto, (self.rect.x + 14, self.rect.y + 58), 16, TEXTO_SUAVE)
            return

        # Ruta anterior en rojo discontinuo y nueva ruta pulsando en amarillo.
        if cambio.anterior:
            for a, b in zip(cambio.anterior, cambio.anterior[1:]):
                linea_discontinua(superficie, ROJO, self.posiciones[a], self.posiciones[b], 2)
        pulso = 0.5 + 0.5 * math.sin(transcurrido * 10)
        self._polilinea(superficie, cambio.nueva, mezclar(AMARILLO, TEXTO, pulso), 4 + int(3 * pulso))

        banda = pygame.Rect(self.rect.x + 10, self.rect.y + 55, self.rect.width - 20, 24)
        pygame.draw.rect(superficie, mezclar(PANEL, AMARILLO, 0.25), banda, border_radius=5)
        self.fuentes.texto(
            superficie,
            f"CAMBIO DE RUTA · {cambio.cluster} ({cambio.motivo}): {texto_ruta(cambio.anterior)}  =>  {texto_ruta(cambio.nueva)}",
            (banda.x + 8, banda.y + 5),
            16,
            AMARILLO,
            negrita=True,
        )

    def _dibujar_paquetes(self, superficie: pygame.Surface) -> None:
        ahora = self.sim.ahora
        for paquete in self.sim.paquetes:
            (x1, y1), (x2, y2) = self.posiciones[paquete.origen], self.posiciones[paquete.destino]
            p = paquete.progreso(ahora)
            color = COLORES_TAREA[self.indice_tipo[paquete.tarea.tipo.nombre] % len(COLORES_TAREA)]
            centro = (int(x1 + (x2 - x1) * p), int(y1 + (y2 - y1) * p))
            pygame.draw.circle(superficie, color, centro, 5)
            pygame.draw.circle(superficie, FONDO, centro, 5, 1)

    def _dibujar_nodos(self, superficie: pygame.Surface, tiempo_real: float) -> None:
        parpadeo = int(tiempo_real * 3) % 2 == 0
        for nodo in self.sim.topologia.nodos.values():
            x, y = self.posiciones[nodo.id]
            if isinstance(nodo, Servidor):
                self._dibujar_servidor(superficie, nodo, x, y, parpadeo)
                continue
            radio = 22 if nodo.tipo is TipoNodo.GATEWAY else 18
            color = COLOR_NODO[nodo.tipo] if nodo.activo else INACTIVO
            pygame.draw.circle(superficie, color, (x, y), radio)
            pygame.draw.circle(superficie, BORDE if nodo.activo else ROJO, (x, y), radio, 2)
            self.fuentes.texto(superficie, nodo.id, (x, y - 7), 17, TEXTO, negrita=True, alineacion="centro")
            if not nodo.activo:
                pygame.draw.line(superficie, ROJO, (x - 12, y - 12), (x + 12, y + 12), 3)
                pygame.draw.line(superficie, ROJO, (x - 12, y + 12), (x + 12, y - 12), 3)
                self.fuentes.texto(superficie, "FALLA", (x, y + radio + 2), 14, ROJO, negrita=True, alineacion="centro")

    def _dibujar_servidor(self, superficie, servidor: Servidor, x: int, y: int, parpadeo: bool) -> None:
        caja = pygame.Rect(0, 0, ANCHO_SERVIDOR, ALTO_SERVIDOR)
        caja.midtop = (x, y - 10)
        emergencia = self.sim.config.termico.temperatura_emergencia
        if not servidor.activo:
            pygame.draw.rect(superficie, INACTIVO, caja, border_radius=6)
            pygame.draw.rect(superficie, ROJO, caja, 2, border_radius=6)
            self.fuentes.texto(superficie, servidor.id, (x, caja.y + 4), 16, TEXTO, negrita=True, alineacion="centro")
            self.fuentes.texto(superficie, "FALLA", (x, caja.y + 24), 15, ROJO, negrita=True, alineacion="centro")
            return

        nivel_termico = (servidor.temperatura - 25) / max(emergencia - 25, 1)
        borde = color_escala(nivel_termico)
        if servidor.en_emergencia:
            borde = ROJO if parpadeo else NARANJA
        pygame.draw.rect(superficie, (30, 36, 44), caja, border_radius=6)
        pygame.draw.rect(superficie, borde, caja, 3 if servidor.en_emergencia else 2, border_radius=6)
        self.fuentes.texto(superficie, servidor.id, (caja.x + 6, caja.y + 4), 15, TEXTO, negrita=True)
        self.fuentes.texto(superficie, f"{servidor.temperatura:.0f}°C", (caja.right - 5, caja.y + 4), 14, borde, alineacion="derecha")

        # Barra de carga: en uso (sólido) + reservado para tareas en tránsito (tenue)
        barra = pygame.Rect(caja.x + 6, caja.y + 22, caja.width - 12, 9)
        pygame.draw.rect(superficie, (48, 54, 61), barra)
        uso = barra.width * servidor.en_uso / servidor.capacidad
        reserva = barra.width * servidor.reservado / servidor.capacidad
        pygame.draw.rect(superficie, mezclar(AZUL, PANEL, 0.5), (barra.x + uso, barra.y, reserva, barra.height))
        pygame.draw.rect(superficie, color_escala(servidor.carga), (barra.x, barra.y, uso, barra.height))
        self.fuentes.texto(
            superficie, f"{servidor.en_uso}/{servidor.capacidad} CPU", (x, caja.y + 34), 13, TEXTO_SUAVE, alineacion="centro"
        )

    def _dibujar_clusters(self, superficie: pygame.Surface) -> None:
        y = self.rect.bottom - 60
        for i, cluster in enumerate(self.sim.clusters):
            color = COLORES_CLUSTER[i % len(COLORES_CLUSTER)]
            seleccionado = i == self.cluster_seleccionado
            marco = self._marco_cluster(i)
            pygame.draw.rect(superficie, color, marco, 2 if seleccionado else 1, border_radius=6)
            titulo = cluster.nombre + ("  [EMERGENCIA]" if cluster.en_emergencia else "")
            self.fuentes.texto(superficie, titulo, (marco.centerx, y), 16, ROJO if cluster.en_emergencia else color,
                               negrita=True, alineacion="centro")
            self.fuentes.texto(
                superficie,
                f"vent {cluster.nivel_ventilador:.0%} · {cluster.energia_asignada:.1f} kW · ×{cluster.factor_velocidad:.2f}",
                (marco.centerx, y + 17),
                14,
                TEXTO_SUAVE,
                alineacion="centro",
            )

    def _dibujar_leyenda(self, superficie: pygame.Surface) -> None:
        x = self.rect.x + 14
        y = self.rect.bottom - 16
        for nombre, indice in self.indice_tipo.items():
            pygame.draw.circle(superficie, COLORES_TAREA[indice % len(COLORES_TAREA)], (x + 5, y + 5), 5)
            x = self.fuentes.texto(superficie, nombre, (x + 14, y - 1), 14, TEXTO_SUAVE).right + 14
        for texto, color in (("enlace libre", color_escala(0)), ("congestionado", color_escala(1)), ("ruta seleccionada", CIAN)):
            pygame.draw.line(superficie, color, (x, y + 5), (x + 18, y + 5), 3)
            x = self.fuentes.texto(superficie, texto, (x + 23, y - 1), 14, TEXTO_SUAVE).right + 14
