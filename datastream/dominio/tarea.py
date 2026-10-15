"""Tareas (paquetes de datos con requerimientos de cómputo)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TipoTarea:
    """Plantilla para generar tareas de una misma clase de servicio."""

    nombre: str
    probabilidad: float
    cpu_min: int
    cpu_max: int
    prioridad_min: int
    prioridad_max: int
    tiempo_servicio: float  # media del tiempo de proceso (s), distribución exponencial


@dataclass
class Tarea:
    id: int
    tipo: TipoTarea
    cpu: int  # "peso" en el modelo de volumen-carga
    prioridad: int  # 1 (baja) a 10 (crítica)
    duracion: float  # tiempo de proceso a velocidad nominal (s)
    llegada: float  # instante de llegada a la red
    reintentos: int = 0
    inicio_proceso: float | None = None

    def espera(self, ahora: float) -> float:
        return ahora - self.llegada

    def valor(self, ahora: float, espera_maxima: float) -> float:
        """Valor para la mochila: trabajo ponderado por prioridad y antigüedad.

        valor = prioridad · cpu · (1 + espera / espera_máxima)
        Multiplicar por cpu hace que la mochila compare prioridad por unidad de
        recurso (si no, siempre preferiría muchas tareas pequeñas). La
        bonificación por espera (hasta duplicar el valor) evita la inanición.
        """
        return self.prioridad * self.cpu * (1 + min(self.espera(ahora) / espera_maxima, 1.0))


@dataclass
class PaqueteEnTransito:
    """Tarea recorriendo un enlace; lo usa el dashboard para animar el tráfico."""

    tarea: Tarea
    origen: str
    destino: str
    inicio: float
    duracion: float

    def progreso(self, ahora: float) -> float:
        if self.duracion <= 0:
            return 1.0
        return min(max((ahora - self.inicio) / self.duracion, 0.0), 1.0)
