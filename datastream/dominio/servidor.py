"""Servidores (nodos finales con capacidad de cómputo) y clústeres."""

from __future__ import annotations

from .red import Nodo, TipoNodo


class Servidor(Nodo):
    """Servidor con capacidad limitada (unidades de CPU/RAM).

    La capacidad se divide en:
        - en_uso   : recursos de tareas que se están procesando.
        - reservado: recursos comprometidos con tareas que aún viajan por la red.
    """

    def __init__(self, identificador: str, etapa: int, capacidad: int, cluster: str):
        super().__init__(identificador, TipoNodo.SERVIDOR, etapa)
        self.capacidad = capacidad
        self.cluster = cluster
        self.en_uso = 0
        self.reservado = 0
        self.procesos: dict[int, object] = {}  # id de tarea → proceso SimPy en curso
        self.completadas = 0
        self.temperatura = 25.0  # °C (la ajusta el modelo térmico)
        self.en_emergencia = False

    @property
    def capacidad_libre(self) -> int:
        if not self.activo:
            return 0
        return self.capacidad - self.en_uso - self.reservado

    @property
    def carga(self) -> float:
        """Fracción de la capacidad ocupada por tareas en proceso (0 a 1)."""
        return self.en_uso / self.capacidad

    @property
    def carga_porcentaje(self) -> float:
        return 100.0 * self.carga

    def reservar(self, cpu: int) -> None:
        if cpu > self.capacidad_libre:
            raise ValueError(f"{self.id}: no hay capacidad para reservar {cpu} unidades.")
        self.reservado += cpu

    def cancelar_reserva(self, cpu: int) -> None:
        self.reservado -= cpu

    def iniciar(self, cpu: int) -> None:
        """La tarea llegó: su reserva pasa a estar en uso."""
        self.reservado -= cpu
        self.en_uso += cpu

    def liberar(self, cpu: int) -> None:
        self.en_uso -= cpu


class Cluster:
    """Grupo de servidores que comparte switch de acceso, refrigeración y energía."""

    def __init__(
        self,
        nombre: str,
        switch: str,
        eficiencia: float,
        latencia_acceso: float,
        servidores: list[Servidor],
    ):
        self.nombre = nombre
        self.switch = switch
        self.eficiencia = eficiencia
        self.latencia_acceso = latencia_acceso
        self.servidores = servidores
        # Variables de control que actualizan los modelos de optimización
        self.nivel_ventilador = 0.5  # método de gradiente
        self.energia_asignada = 0.0  # Lagrange
        self.factor_velocidad = 1.0  # depende de la energía asignada
        self.temperatura_max = 25.0  # modelo térmico
        self.en_emergencia = False

    @property
    def capacidad_total(self) -> int:
        return sum(s.capacidad for s in self.servidores)

    @property
    def capacidad_activa(self) -> int:
        return sum(s.capacidad for s in self.servidores if s.activo)

    @property
    def carga(self) -> float:
        """Recursos en uso sobre la capacidad instalada del clúster (0 a 1)."""
        return sum(s.en_uso for s in self.servidores) / self.capacidad_total

    @property
    def servidores_activos(self) -> int:
        return sum(1 for s in self.servidores if s.activo)
