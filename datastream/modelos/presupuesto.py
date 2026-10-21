"""Módulo 2 · Asignación del presupuesto eléctrico con multiplicadores de Lagrange.

El centro de datos dispone de una energía total E_max (kW) que se reparte
entre n clústeres. El rendimiento de cada clúster tiene rendimientos
decrecientes respecto a la energía que recibe:

    R_i(E_i) = a_i · √E_i        (a_i = coeficiente de rendimiento del clúster)

Problema con restricción de igualdad:

    max  R(E_1, …, E_n) = Σ a_i·√E_i
    s.a. g(E) = Σ E_i − E_max = 0

Función de Lagrange:  L(E, λ) = Σ a_i·√E_i − λ·(Σ E_i − E_max)

Condiciones de primer orden (derivadas parciales igualadas a cero):

    ∂L/∂E_i = a_i / (2·√E_i) − λ = 0   ⇒   E_i = a_i² / (4·λ²)
    ∂L/∂λ   = −(Σ E_i − E_max) = 0     ⇒   Σ a_i² / (4·λ²) = E_max

    ⇒   λ* = √(Σ a_i²) / (2·√E_max)     y     E_i* = E_max · a_i² / Σ a_j²

λ* es el "precio sombra": cuánto rendimiento adicional se obtendría con un kW
más de presupuesto. Como ∂²R/∂E_i² = −a_i / (4·E_i^(3/2)) < 0, R es cóncava y
el punto encontrado es un máximo.

Condición de segundo orden con el Hessiano orlado (n variables, 1 restricción):

        | 0    1     1    …   1   |
    H = | 1   L_11   0    …   0   |      L_ii = ∂²L/∂E_i² = −a_i / (4·E_i^(3/2))
        | 1    0    L_22  …   0   |
        | …                       |

Si los menores principales de orden 3, 4, …, n+1 alternan de signo empezando
por positivo (+, −, +, …), el punto estacionario es un máximo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class ResultadoLagrange:
    energias: tuple[float, ...]
    multiplicador: float  # λ*
    rendimientos: tuple[float, ...]
    rendimiento_total: float
    presupuesto: float


class AsignadorEnergia:
    def __init__(self, presupuesto_total: float):
        if presupuesto_total <= 0:
            raise ValueError("El presupuesto eléctrico debe ser positivo.")
        self.presupuesto = float(presupuesto_total)

    def asignar(self, coeficientes: Sequence[float]) -> ResultadoLagrange:
        """Solución analítica obtenida con las condiciones de Lagrange."""
        if not coeficientes:
            raise ValueError("Se requiere al menos un clúster.")
        if any(a < 0 for a in coeficientes):
            raise ValueError("Los coeficientes de rendimiento no pueden ser negativos.")

        suma_cuadrados = sum(a * a for a in coeficientes)
        if suma_cuadrados == 0:  # ningún clúster puede aprovechar energía
            ceros = tuple(0.0 for _ in coeficientes)
            return ResultadoLagrange(ceros, 0.0, ceros, 0.0, self.presupuesto)

        multiplicador = math.sqrt(suma_cuadrados) / (2 * math.sqrt(self.presupuesto))
        energias = tuple(self.presupuesto * a * a / suma_cuadrados for a in coeficientes)
        rendimientos = tuple(a * math.sqrt(e) for a, e in zip(coeficientes, energias))
        return ResultadoLagrange(energias, multiplicador, rendimientos, sum(rendimientos), self.presupuesto)

    @staticmethod
    def verificar(resultado: ResultadoLagrange, coeficientes: Sequence[float]) -> dict[str, float]:
        """Evalúa las condiciones de primer orden en la solución (deben ser ≈ 0)."""
        residuos = [
            a / (2 * math.sqrt(e)) - resultado.multiplicador
            for a, e in zip(coeficientes, resultado.energias)
            if e > 0
        ]
        return {
            "max_residuo_dL_dEi": max((abs(r) for r in residuos), default=0.0),
            "residuo_restriccion": sum(resultado.energias) - resultado.presupuesto,
        }

    @staticmethod
    def hessiano_orlado(resultado: ResultadoLagrange, coeficientes: Sequence[float]) -> dict[str, object]:
        """Condición de segundo orden: Hessiano orlado y sus menores principales.

        Solo se consideran los clústeres con energía asignada (E_i > 0), porque
        en E_i = 0 la segunda derivada no está definida.
        """
        segundas = [-a / (4 * e**1.5) for a, e in zip(coeficientes, resultado.energias) if e > 0]
        n = len(segundas)
        matriz = np.zeros((n + 1, n + 1))
        matriz[0, 1:] = matriz[1:, 0] = 1.0  # gradiente de la restricción Σ E_i = E_max
        for i, valor in enumerate(segundas, start=1):
            matriz[i, i] = valor
        menores = [float(np.linalg.det(matriz[:orden, :orden])) for orden in range(3, n + 2)]
        # Máximo: el menor de orden k debe tener el signo de (−1)^(k−1), es decir +, −, +, …
        es_maximo = all(m * (-1) ** (orden - 1) > 0 for orden, m in zip(range(3, n + 2), menores))
        return {"matriz": matriz, "menores": menores, "es_maximo": es_maximo}

    @staticmethod
    def derivacion_simbolica(n: int) -> dict[str, object]:
        """Construye el Lagrangiano con sympy y devuelve sus derivadas parciales."""
        if n < 1:
            raise ValueError("Se requiere al menos un clúster.")
        energias = sp.symbols(f"E1:{n + 1}", positive=True)
        coeficientes = sp.symbols(f"a1:{n + 1}", positive=True)
        lam, e_max = sp.symbols("lambda E_max", positive=True)
        lagrangiano = sum(a * sp.sqrt(e) for a, e in zip(coeficientes, energias)) - lam * (
            sum(energias) - e_max
        )
        parciales = [sp.Eq(sp.diff(lagrangiano, e), 0) for e in energias]
        parciales.append(sp.Eq(sp.diff(lagrangiano, lam), 0))
        return {"lagrangiano": lagrangiano, "condiciones": parciales}
