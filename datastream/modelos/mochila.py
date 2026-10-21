"""Módulo 1 · Modelo de volumen-carga (problema de la mochila 0/1).

Cada servidor tiene una capacidad máxima W (unidades de CPU/RAM) y cada tarea
que espera en la cola es un "artículo" con peso w_i (recursos que requiere) y
valor v_i (prioridad, con bonificación por tiempo de espera).

Formulación por programación dinámica:
    - Etapa i  : decidir si la tarea i entra (m_i = 1) o no (m_i = 0).
    - Estado w : capacidad todavía disponible en el servidor.
    - Recursión:
          f_0(w) = 0
          f_i(w) = f_{i-1}(w)                                  si w_i > w
          f_i(w) = max{ f_{i-1}(w), v_i + f_{i-1}(w - w_i) }    si w_i <= w

f_n(W) es el valor máximo que puede procesar el servidor sin exceder su
capacidad; las tareas elegidas se recuperan recorriendo la tabla hacia atrás.

En la simulación cada tarea es única, por eso se usa la versión 0/1. El modelo
general visto en clase permite varias unidades de cada artículo
(m_i = 0, 1, …, ⌊W/w_i⌋) y se resuelve con la recursión hacia atrás:

    f_{n+1}(x) = 0
    f_i(x_i) = max { r_i·m_i + f_{i+1}(x_i − w_i·m_i) },   m_i = 0, 1, …, ⌊x_i/w_i⌋

La versión 0/1 es el caso particular m_i ∈ {0, 1} (ModeloMochilaUnidades).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Articulo:
    """Carga candidata a entrar en la mochila (una tarea de la cola)."""

    nombre: str
    peso: int
    valor: float


@dataclass(frozen=True)
class ResultadoMochila:
    capacidad: int
    seleccionados: tuple[int, ...]  # índices de los artículos elegidos
    valor_total: float
    peso_total: int
    tabla: tuple[tuple[float, ...], ...] | None = None  # f_i(w), solo si se pidió

    @property
    def utilizacion(self) -> float:
        """Fracción de la capacidad del servidor que queda ocupada."""
        return self.peso_total / self.capacidad if self.capacidad else 0.0


class ModeloVolumenCarga:
    """Resuelve el modelo de volumen-carga con programación dinámica."""

    def resolver(
        self, articulos: Sequence[Articulo], capacidad: int, guardar_tabla: bool = False
    ) -> ResultadoMochila:
        self._validar(articulos, capacidad)
        n = len(articulos)

        # f[i][w]: mejor valor usando los primeros i artículos con capacidad w.
        f = [[0.0] * (capacidad + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            peso, valor = articulos[i - 1].peso, articulos[i - 1].valor
            fila_anterior, fila = f[i - 1], f[i]
            for w in range(capacidad + 1):
                no_tomar = fila_anterior[w]
                if peso <= w:
                    tomar = valor + fila_anterior[w - peso]
                    fila[w] = tomar if tomar > no_tomar else no_tomar
                else:
                    fila[w] = no_tomar

        # Reconstrucción de la decisión óptima (recorrido hacia atrás).
        seleccionados = []
        w = capacidad
        for i in range(n, 0, -1):
            if f[i][w] != f[i - 1][w]:  # el artículo i fue tomado
                seleccionados.append(i - 1)
                w -= articulos[i - 1].peso
        seleccionados.reverse()

        tabla = tuple(tuple(fila) for fila in f) if guardar_tabla else None
        return ResultadoMochila(
            capacidad=capacidad,
            seleccionados=tuple(seleccionados),
            valor_total=f[n][capacidad],
            peso_total=sum(articulos[i].peso for i in seleccionados),
            tabla=tabla,
        )

    @staticmethod
    def _validar(articulos: Sequence[Articulo], capacidad: int) -> None:
        if not isinstance(capacidad, int) or capacidad < 0:
            raise ValueError("La capacidad debe ser un entero no negativo.")
        for articulo in articulos:
            if not isinstance(articulo.peso, int) or articulo.peso <= 0:
                raise ValueError(f"El peso de {articulo.nombre} debe ser un entero positivo.")
            if articulo.valor < 0:
                raise ValueError(f"El valor de {articulo.nombre} no puede ser negativo.")


@dataclass(frozen=True)
class FilaMochila:
    """Una fila de la tabla de la etapa i: estado x_i y valor de cada alternativa m_i."""

    estado: int  # x_i: capacidad disponible para los artículos i, i+1, …, n
    alternativas: tuple[float | None, ...]  # r_i·m_i + f_{i+1}(x_i − w_i·m_i) para m_i = 0, 1, … (None si no cabe)
    valor_optimo: float  # f_i(x_i)
    decision: int  # m_i*


@dataclass(frozen=True)
class EtapaMochila:
    articulo: Articulo
    maximo_unidades: int  # ⌊W / w_i⌋
    filas: tuple[FilaMochila, ...]


@dataclass(frozen=True)
class ResultadoMochilaUnidades:
    capacidad: int
    cantidades: tuple[int, ...]  # m_i* de cada artículo
    valor_total: float
    peso_total: int
    etapas: tuple[EtapaMochila, ...]  # en orden de cálculo: del último artículo al primero


class ModeloMochilaUnidades:
    """Modelo de la mochila con varias unidades por artículo (recursión hacia atrás).

    `limites` permite acotar las unidades de cada artículo; con límite 1 para
    todos se obtiene la mochila 0/1.
    """

    def resolver(
        self, articulos: Sequence[Articulo], capacidad: int, limites: Sequence[int] | None = None
    ) -> ResultadoMochilaUnidades:
        ModeloVolumenCarga._validar(articulos, capacidad)
        n = len(articulos)
        if limites is not None and (len(limites) != n or any(l < 0 for l in limites)):
            raise ValueError("Se requiere un límite de unidades no negativo por artículo.")

        siguiente = [0.0] * (capacidad + 1)  # f_{n+1}(x) = 0 para todo x
        decisiones: list[list[int]] = [[] for _ in range(n)]
        etapas = []
        for i in range(n - 1, -1, -1):
            articulo = articulos[i]
            maximo = capacidad // articulo.peso
            if limites is not None:
                maximo = min(maximo, limites[i])
            actual, filas = [], []
            for x in range(capacidad + 1):
                alternativas = tuple(
                    articulo.valor * m + siguiente[x - articulo.peso * m] if articulo.peso * m <= x else None
                    for m in range(maximo + 1)
                )
                mejor = max(range(len(alternativas)), key=lambda m: (alternativas[m] is not None, alternativas[m] or 0.0))
                actual.append(alternativas[mejor])
                decisiones[i].append(mejor)
                filas.append(FilaMochila(x, alternativas, alternativas[mejor], mejor))
            etapas.append(EtapaMochila(articulo, maximo, tuple(filas)))
            siguiente = actual

        # Política óptima: desde el estado inicial x_1 = W hacia adelante.
        cantidades, x = [], capacidad
        for i in range(n):
            m = decisiones[i][x]
            cantidades.append(m)
            x -= articulos[i].peso * m
        return ResultadoMochilaUnidades(
            capacidad=capacidad,
            cantidades=tuple(cantidades),
            valor_total=siguiente[capacidad] if n else 0.0,
            peso_total=sum(a.peso * m for a, m in zip(articulos, cantidades)),
            etapas=tuple(etapas),
        )
