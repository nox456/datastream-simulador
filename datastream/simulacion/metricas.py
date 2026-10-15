"""Recolección de métricas y eventos durante la simulación."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Muestra:
    """Fotografía del sistema tomada en cada ciclo de control."""

    t: float
    en_cola: int
    en_transito: int
    en_proceso: int
    utilizacion: float  # recursos en uso / capacidad instalada
    throughput: float  # tareas completadas por segundo en el último intervalo
    tasa_llegada: float  # λ(t) instantánea
    niveles_ventilador: tuple[float, ...]
    energia_refrigeracion: float
    iteraciones_gradiente: int
    energias_cluster: tuple[float, ...]
    multiplicador: float  # λ* de Lagrange
    rendimiento_total: float
    temperaturas: tuple[float, ...]  # temperatura máxima por clúster

    @property
    def en_sistema(self) -> int:
        return self.en_cola + self.en_transito + self.en_proceso


@dataclass(frozen=True)
class Evento:
    t: float
    tipo: str  # falla | reparacion | ruta | emergencia | info
    mensaje: str


class Metricas:
    def __init__(self) -> None:
        self.generadas = 0
        self.procesadas = 0
        self.descartadas = 0  # superaron la espera máxima en cola
        self.retransmitidas = 0  # paquetes perdidos por fallas de nodos
        self.interrumpidas = 0  # tareas reiniciadas por caída de su servidor
        self.fallas = 0
        self.cambios_ruta = 0
        self.ejecuciones_mochila = 0
        self.suma_utilizacion_mochila = 0.0
        self.valor_procesado = 0.0
        self.suma_espera = 0.0  # llegada → inicio de proceso
        self.suma_sistema = 0.0  # llegada → fin de proceso
        self.muestras: list[Muestra] = []
        self.eventos: deque[Evento] = deque(maxlen=400)
        self.completadas_por_tipo: dict[str, int] = {}
        self.descartadas_por_tipo: dict[str, int] = {}

    def registrar_evento(self, t: float, tipo: str, mensaje: str) -> None:
        self.eventos.append(Evento(t, tipo, mensaje))

    def registrar_mochila(self, utilizacion: float) -> None:
        self.ejecuciones_mochila += 1
        self.suma_utilizacion_mochila += utilizacion

    def registrar_descarte(self, tarea) -> None:
        self.descartadas += 1
        nombre = tarea.tipo.nombre
        self.descartadas_por_tipo[nombre] = self.descartadas_por_tipo.get(nombre, 0) + 1

    def registrar_completada(self, tarea, ahora: float, valor: float) -> None:
        self.procesadas += 1
        self.valor_procesado += valor
        self.suma_espera += tarea.inicio_proceso - tarea.llegada
        self.suma_sistema += ahora - tarea.llegada
        nombre = tarea.tipo.nombre
        self.completadas_por_tipo[nombre] = self.completadas_por_tipo.get(nombre, 0) + 1

    @property
    def espera_promedio(self) -> float:
        return self.suma_espera / self.procesadas if self.procesadas else 0.0

    @property
    def tiempo_sistema_promedio(self) -> float:
        return self.suma_sistema / self.procesadas if self.procesadas else 0.0

    @property
    def utilizacion_mochila_promedio(self) -> float:
        return self.suma_utilizacion_mochila / self.ejecuciones_mochila if self.ejecuciones_mochila else 0.0

    def promedio(self, atributo: str) -> float:
        if not self.muestras:
            return 0.0
        return sum(getattr(m, atributo) for m in self.muestras) / len(self.muestras)

    def serie(self, atributo: str, desde: float = 0.0) -> tuple[list[float], list]:
        """Devuelve (tiempos, valores) de un atributo de las muestras a partir de `desde`."""
        tiempos, valores = [], []
        for muestra in self.muestras:
            if muestra.t >= desde:
                tiempos.append(muestra.t)
                valores.append(getattr(muestra, atributo))
        return tiempos, valores
