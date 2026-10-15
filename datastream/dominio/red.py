"""Entidades de la red: nodos (routers/switches), enlaces y topología por etapas."""

from __future__ import annotations

from enum import Enum


class TipoNodo(Enum):
    GATEWAY = "gateway"
    NUCLEO = "núcleo"
    AGREGACION = "agregación"
    ACCESO = "acceso (ToR)"
    SERVIDOR = "servidor"


class Nodo:
    """Elemento de la red que puede fallar y repararse."""

    def __init__(self, identificador: str, tipo: TipoNodo, etapa: int):
        self.id = identificador
        self.tipo = tipo
        self.etapa = etapa
        self.activo = True
        self.fallas = 0

    def fallar(self) -> None:
        self.activo = False
        self.fallas += 1

    def reparar(self) -> None:
        self.activo = True

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id})"


class Enlace:
    """Enlace dirigido entre dos nodos de etapas consecutivas.

    La latencia crece linealmente con los paquetes que lo atraviesan:
        latencia = latencia_base · (1 + factor_congestion · ocupación)
    Para enrutar se usa la ocupación suavizada (promedio móvil exponencial),
    que evita que las rutas oscilen con cada paquete individual.
    """

    def __init__(self, origen: str, destino: str, latencia_base: float):
        self.origen = origen
        self.destino = destino
        self.latencia_base = latencia_base
        self.en_transito = 0  # paquetes viajando ahora mismo por el enlace
        self.congestion = 0.0  # ocupación suavizada (EWMA)
        self.paquetes_totales = 0

    def actualizar_congestion(self, suavizado: float) -> None:
        self.congestion = (1 - suavizado) * self.congestion + suavizado * self.en_transito

    def latencia_estimada(self, factor_congestion: float) -> float:
        """Latencia que ve el algoritmo de enrutamiento."""
        return self.latencia_base * (1 + factor_congestion * self.congestion)

    def tiempo_transito(self, factor_congestion: float) -> float:
        """Tiempo real que tarda un paquete que entra ahora al enlace."""
        return self.latencia_base * (1 + factor_congestion * self.en_transito)


class Topologia:
    """Red organizada en etapas (la última etapa son los servidores)."""

    def __init__(self, etapas: list[list[Nodo]], enlaces: list[Enlace]):
        self.etapas = etapas
        self.nodos: dict[str, Nodo] = {nodo.id: nodo for etapa in etapas for nodo in etapa}
        self.enlaces: dict[tuple[str, str], Enlace] = {(e.origen, e.destino): e for e in enlaces}

    @property
    def gateway(self) -> Nodo:
        return self.etapas[0][0]

    def ids_por_etapa(self) -> list[list[str]]:
        return [[nodo.id for nodo in etapa] for etapa in self.etapas]

    def enlace(self, origen: str, destino: str) -> Enlace:
        return self.enlaces[(origen, destino)]

    def routers(self) -> list[Nodo]:
        """Nodos de red (sin servidores), en orden de etapa."""
        return [nodo for etapa in self.etapas[:-1] for nodo in etapa]

    def arcos_disponibles(self, factor_congestion: float) -> dict[tuple[str, str], float]:
        """Enlaces utilizables (ambos extremos activos) con su latencia estimada."""
        return {
            clave: enlace.latencia_estimada(factor_congestion)
            for clave, enlace in self.enlaces.items()
            if self.nodos[enlace.origen].activo and self.nodos[enlace.destino].activo
        }
