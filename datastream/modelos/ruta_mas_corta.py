"""Módulo 1 · Ruta más corta mediante programación dinámica por etapas.

La red del centro de datos está organizada en etapas (gateway → núcleo →
agregación → acceso/ToR → servidores), igual que el clásico "problema de la
diligencia" (stagecoach problem). Se aplica la recursión hacia adelante:

    f_0(origen) = 0
    f_k(y) = min { f_{k-1}(x) + d(x, y) }   para todo nodo x de la etapa k-1

donde:
    - etapa k     : nivel de la red que se está evaluando.
    - estado y    : nodo (router o servidor) de la etapa k.
    - d(x, y)     : latencia actual del enlace x → y (crece con la congestión).
    - f_k(y)      : latencia mínima acumulada desde el origen hasta y.

La política óptima (la ruta) se reconstruye siguiendo las decisiones guardadas
(predecesores) desde el destino hacia el origen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INFINITO = math.inf


@dataclass(frozen=True)
class Candidato:
    """Alternativa evaluada en la recursión: llegar a y desde x."""

    desde: str
    costo: float  # f_{k-1}(x) + d(x, y)


@dataclass(frozen=True)
class FilaEtapa:
    """Una fila de la tabla de programación dinámica (etapa k, estado y)."""

    etapa: int
    nodo: str
    candidatos: tuple[Candidato, ...]
    valor_optimo: float  # f_k(y)
    decision: str | None  # x* que minimiza la recursión


@dataclass
class ResultadoRutas:
    """Resultado de resolver la recursión para todos los nodos de la red."""

    origen: str
    distancias: dict[str, float]
    predecesores: dict[str, str | None]
    tabla: list[FilaEtapa] = field(default_factory=list)

    def es_alcanzable(self, destino: str) -> bool:
        return self.distancias.get(destino, INFINITO) < INFINITO

    def latencia_hacia(self, destino: str) -> float:
        return self.distancias.get(destino, INFINITO)

    def ruta_hacia(self, destino: str) -> list[str] | None:
        """Reconstruye la ruta óptima origen → destino (None si no existe)."""
        if not self.es_alcanzable(destino):
            return None
        ruta = [destino]
        while ruta[-1] != self.origen:
            anterior = self.predecesores.get(ruta[-1])
            if anterior is None:
                return None
            ruta.append(anterior)
        ruta.reverse()
        return ruta


class RutaMasCortaDP:
    """Resuelve la ruta de menor latencia con programación dinámica por etapas."""

    def __init__(self, etapas: list[list[str]]):
        if not etapas or not all(etapas):
            raise ValueError("La red debe tener al menos una etapa y ninguna etapa vacía.")
        if len(etapas[0]) != 1:
            raise ValueError("La primera etapa debe contener un único nodo de origen.")
        todos = [nodo for etapa in etapas for nodo in etapa]
        if len(todos) != len(set(todos)):
            raise ValueError("Un nodo no puede aparecer en más de una etapa.")
        self.etapas = [list(etapa) for etapa in etapas]

    @property
    def origen(self) -> str:
        return self.etapas[0][0]

    def resolver(self, arcos: dict[tuple[str, str], float]) -> ResultadoRutas:
        """Aplica la recursión hacia adelante.

        `arcos` contiene únicamente los enlaces utilizables en este instante
        (los enlaces de nodos caídos se omiten) con su latencia actual.
        """
        for (origen, destino), latencia in arcos.items():
            if latencia < 0:
                raise ValueError(f"Latencia negativa en el enlace {origen}→{destino}.")

        distancias: dict[str, float] = {self.origen: 0.0}  # f_0(origen) = 0
        predecesores: dict[str, str | None] = {self.origen: None}
        tabla: list[FilaEtapa] = []

        for k in range(1, len(self.etapas)):
            for y in self.etapas[k]:
                candidatos = []
                for x in self.etapas[k - 1]:
                    latencia = arcos.get((x, y))
                    f_anterior = distancias.get(x, INFINITO)
                    if latencia is None or f_anterior == INFINITO:
                        continue  # no hay enlace o x no es alcanzable
                    candidatos.append(Candidato(x, f_anterior + latencia))

                if candidatos:
                    mejor = min(candidatos, key=lambda c: c.costo)
                    distancias[y], predecesores[y] = mejor.costo, mejor.desde
                else:
                    distancias[y], predecesores[y] = INFINITO, None

                tabla.append(FilaEtapa(k, y, tuple(candidatos), distancias[y], predecesores[y]))

        return ResultadoRutas(self.origen, distancias, predecesores, tabla)
