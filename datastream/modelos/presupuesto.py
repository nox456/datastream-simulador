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
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
import sympy as sp
from scipy.optimize import minimize

METODOS_LAGRANGE = ("analitico", "simbolico", "numerico")


@dataclass(frozen=True)
class ResultadoLagrange:
    energias: tuple[float, ...]
    multiplicador: float  # λ*
    rendimientos: tuple[float, ...]
    rendimiento_total: float
    presupuesto: float
    metodo: str = "analitico"


class AsignadorEnergia:
    """Reparto del presupuesto eléctrico por uno de tres métodos que deben dar el mismo óptimo:

    - analitico: la fórmula E_i* = E_max·a_i²/Σa_j² despejada a mano de las condiciones de Lagrange.
    - simbolico: sympy resuelve el sistema ∇L = 0 con coeficientes simbólicos (una vez por cantidad
      de clústeres) y esa solución general se evalúa con los coeficientes de cada instante.
    - numerico: SciPy maximiza R(E) directamente con la restricción Σ E_i = E_max (método SLSQP).
    """

    def __init__(self, presupuesto_total: float, metodo: str = "analitico"):
        if presupuesto_total <= 0:
            raise ValueError("El presupuesto eléctrico debe ser positivo.")
        if metodo not in METODOS_LAGRANGE:
            raise ValueError(f"Método de Lagrange desconocido: {metodo!r} (use {', '.join(METODOS_LAGRANGE)}).")
        self.presupuesto = float(presupuesto_total)
        self.metodo = metodo

    def asignar(self, coeficientes: Sequence[float]) -> ResultadoLagrange:
        """Reparto óptimo según el método elegido."""
        if not coeficientes:
            raise ValueError("Se requiere al menos un clúster.")
        if any(a < 0 for a in coeficientes):
            raise ValueError("Los coeficientes de rendimiento no pueden ser negativos.")

        suma_cuadrados = sum(a * a for a in coeficientes)
        if suma_cuadrados == 0:  # ningún clúster puede aprovechar energía
            ceros = tuple(0.0 for _ in coeficientes)
            return ResultadoLagrange(ceros, 0.0, ceros, 0.0, self.presupuesto, self.metodo)

        if self.metodo == "simbolico":
            energias, multiplicador = self._por_solucion_simbolica(coeficientes)
        elif self.metodo == "numerico":
            energias, multiplicador = self._por_optimizacion_numerica(coeficientes)
        else:
            multiplicador = math.sqrt(suma_cuadrados) / (2 * math.sqrt(self.presupuesto))
            energias = tuple(self.presupuesto * a * a / suma_cuadrados for a in coeficientes)
        rendimientos = tuple(a * math.sqrt(e) for a, e in zip(coeficientes, energias))
        return ResultadoLagrange(energias, multiplicador, rendimientos, sum(rendimientos), self.presupuesto, self.metodo)

    def _por_solucion_simbolica(self, coeficientes: Sequence[float]) -> tuple[tuple[float, ...], float]:
        evaluar = _solucion_general_evaluable(len(coeficientes))
        *energias, multiplicador = evaluar(*coeficientes, self.presupuesto)
        return tuple(float(e) for e in energias), float(multiplicador)

    def _por_optimizacion_numerica(self, coeficientes: Sequence[float]) -> tuple[tuple[float, ...], float]:
        a, total, n = np.asarray(coeficientes, float), self.presupuesto, len(coeficientes)
        resultado = minimize(
            lambda e: -np.sum(a * np.sqrt(np.maximum(e, 0.0))),
            np.full(n, total / n),
            jac=lambda e: -a / (2 * np.sqrt(np.maximum(e, 1e-12))),
            method="SLSQP",
            bounds=[(0.0, total)] * n,
            constraints=[{"type": "eq", "fun": lambda e: e.sum() - total, "jac": lambda e: np.ones(n)}],
            options={"ftol": 1e-12, "maxiter": 200},
        )
        if not resultado.success:
            raise RuntimeError(f"SciPy no encontró el óptimo: {resultado.message}")
        energias = np.clip(resultado.x, 0.0, None)
        energias = energias * total / energias.sum()
        activos = energias > 1e-9 * total
        multiplicador = float(np.mean(a[activos] / (2 * np.sqrt(energias[activos]))))
        return tuple(float(e) for e in energias), multiplicador

    @staticmethod
    def solucion_simbolica(n: int) -> dict[sp.Symbol, sp.Expr]:
        """Solución general del sistema ∇L = 0 obtenida con sp.solve (E_1*, …, E_n* y λ*)."""
        return _solucion_general(n)

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


@lru_cache(maxsize=None)
def _solucion_general(n: int) -> dict[sp.Symbol, sp.Expr]:
    """sp.solve sobre las condiciones de primer orden con coeficientes simbólicos (tarda unos segundos)."""
    if n < 1:
        raise ValueError("Se requiere al menos un clúster.")
    simbolico = AsignadorEnergia.derivacion_simbolica(n)
    condiciones = simbolico["condiciones"]
    incognitas = [sp.Symbol(f"E{i}", positive=True) for i in range(1, n + 1)] + [sp.Symbol("lambda", positive=True)]
    soluciones = sp.solve(condiciones, incognitas, dict=True)
    if len(soluciones) != 1:
        raise RuntimeError(f"sympy encontró {len(soluciones)} soluciones; se esperaba una única.")
    return soluciones[0]


@lru_cache(maxsize=None)
def _solucion_general_evaluable(n: int) -> Callable[..., list[float]]:
    """Convierte la solución general en una función numérica f(a_1, …, a_n, E_max) → [E_1*, …, E_n*, λ*]."""
    solucion = _solucion_general(n)
    coeficientes = sp.symbols(f"a1:{n + 1}", positive=True)
    e_max = sp.Symbol("E_max", positive=True)
    incognitas = [sp.Symbol(f"E{i}", positive=True) for i in range(1, n + 1)] + [sp.Symbol("lambda", positive=True)]
    return sp.lambdify([*coeficientes, e_max], [solucion[x] for x in incognitas], "math")
