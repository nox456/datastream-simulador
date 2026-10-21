"""Módulo 1 · Ruta más corta en un grafo general con Bellman-Ford (programación dinámica).

La DP por etapas exige que la red esté organizada en etapas y que cada enlace
una etapas consecutivas. Bellman-Ford no tiene esa restricción: acepta enlaces
dentro de una misma etapa, ciclos y hasta latencias negativas. Es también
programación dinámica, con el número de enlaces usados como etapa:

    f_0(origen) = 0,   f_0(v) = ∞ para v ≠ origen
    f_k(v) = min { f_{k-1}(v),  min_u [ f_{k-1}(u) + d(u, v) ] }

f_k(v) es la latencia mínima desde el origen hasta v usando a lo sumo k
enlaces. Un camino simple tiene como máximo n − 1 enlaces, así que f_{n-1} es
la solución. Si en la iteración n todavía se puede mejorar algún valor, hay un
ciclo de latencia negativa y la ruta más corta no está definida.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ruta_mas_corta import INFINITO, Candidato, FilaEtapa, ResultadoRutas


@dataclass
class ResultadoBellmanFord(ResultadoRutas):
    """Resultado compatible con el de la DP por etapas, más las iteraciones f_k."""

    iteraciones: list[dict[str, float]] = field(default_factory=list)
    ciclo_negativo: bool = False

    def ruta_hacia(self, destino: str) -> list[str] | None:
        if self.ciclo_negativo:
            return None
        ruta = super().ruta_hacia(destino)
        return ruta if ruta is None or len(ruta) <= len(self.distancias) else None


class BellmanFordDP:
    """Ruta de menor latencia desde un origen hacia todos los nodos de un grafo dirigido general."""

    def __init__(self, nodos: list[str], origen: str):
        if len(nodos) != len(set(nodos)):
            raise ValueError("Los nodos del grafo no pueden repetirse.")
        if origen not in nodos:
            raise ValueError(f"El origen '{origen}' no pertenece al grafo.")
        self.nodos = list(nodos)
        self.origen = origen

    def resolver(self, arcos: dict[tuple[str, str], float]) -> ResultadoBellmanFord:
        """`arcos` contiene los enlaces utilizables (los de nodos caídos se omiten) con su latencia."""
        for (u, v) in arcos:
            if u not in self.nodos or v not in self.nodos:
                raise ValueError(f"El enlace {u}→{v} usa un nodo que no pertenece al grafo.")
            if u == v:
                raise ValueError(f"El enlace {u}→{v} es un lazo sobre el mismo nodo.")

        f = {v: INFINITO for v in self.nodos}
        f[self.origen] = 0.0
        predecesores: dict[str, str | None] = {v: None for v in self.nodos}
        iteraciones = [dict(f)]

        for _ in range(len(self.nodos) - 1):
            anterior, cambio = dict(f), False
            for (u, v), latencia in arcos.items():
                if anterior[u] + latencia < f[v]:
                    f[v], predecesores[v], cambio = anterior[u] + latencia, u, True
            if not cambio:
                break
            iteraciones.append(dict(f))

        ciclo_negativo = any(f[u] + latencia < f[v] for (u, v), latencia in arcos.items() if f[u] < INFINITO)

        resultado = ResultadoBellmanFord(self.origen, f, predecesores, [], iteraciones, ciclo_negativo)
        resultado.tabla = self._tabla(arcos, resultado)
        return resultado

    def _tabla(self, arcos: dict[tuple[str, str], float], resultado: ResultadoBellmanFord) -> list[FilaEtapa]:
        """Una fila por nodo con todos sus candidatos; la 'etapa' es el número de enlaces de su ruta óptima."""
        f = resultado.distancias
        filas = []
        for v in self.nodos:
            if v == self.origen:
                continue
            candidatos = tuple(Candidato(u, f[u] + d) for (u, w), d in arcos.items() if w == v and f[u] < INFINITO)
            ruta = resultado.ruta_hacia(v)
            enlaces = len(ruta) - 1 if ruta else 0
            filas.append(FilaEtapa(enlaces, v, candidatos, f[v], resultado.predecesores[v]))
        return sorted(filas, key=lambda fila: (fila.etapa == 0, fila.etapa))
