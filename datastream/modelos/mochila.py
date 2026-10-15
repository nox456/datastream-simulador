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
