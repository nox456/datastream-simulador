"""Módulo 2 · Ajuste dinámico de refrigeración con el método de gradiente.

Variables: u_i ∈ [u_min, 1] = potencia relativa de los ventiladores del clúster i.
Datos:     L_i ∈ [0, 1]      = carga de trabajo actual del clúster i.

Consumo energético total (kW) que se quiere minimizar:

    E(u) = Σ_i [ A·u_i³  +  B·L_i² / u_i ]  +  C·Σ_(i,j vecinos) (u_i − u_j)²

    - A·u_i³        : potencia de los ventiladores (ley de afinidad: P ∝ velocidad³).
    - B·L_i²/u_i    : sobreconsumo por calor no evacuado (fugas de corriente y
                      trabajo extra del chiller); baja cuando hay más flujo de aire.
    - C·(u_i − u_j)²: penalización por desbalance de presión entre pasillos vecinos.

Derivadas parciales (gradiente):

    ∂E/∂u_i = 3A·u_i² − B·L_i²/u_i² + 2C·Σ_{j∈N(i)} (u_i − u_j)

Método de gradiente (descenso más pronunciado con proyección a los límites):

    u^(k+1) = Proy_[u_min, 1]( u^(k) − α·∇E(u^(k)) )

Se detiene cuando el paso es menor que la tolerancia. En la simulación se parte
de la solución anterior (arranque en caliente), de modo que los ventiladores se
ajustan continuamente a medida que cambia la carga.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ResultadoGradiente:
    niveles: tuple[float, ...]
    energia: float
    iteraciones: int
    convergio: bool
    norma_gradiente: float
    historial_energia: tuple[float, ...] = field(default=())  # E(u^(k)) por iteración
    historial_niveles: tuple[tuple[float, ...], ...] = field(default=())  # u^(k) por iteración


class OptimizadorRefrigeracion:
    def __init__(
        self,
        coef_ventilador: float,
        coef_calor: float,
        coef_acople: float,
        vecinos: Sequence[tuple[int, int]],
        tasa_aprendizaje: float = 0.02,
        tolerancia: float = 1e-4,
        max_iteraciones: int = 200,
        nivel_minimo: float = 0.1,
    ):
        if min(coef_ventilador, coef_calor) <= 0 or coef_acople < 0:
            raise ValueError("Los coeficientes A y B deben ser positivos y C no negativo.")
        if tasa_aprendizaje <= 0 or tolerancia <= 0 or max_iteraciones < 1:
            raise ValueError("Tasa de aprendizaje, tolerancia y máximo de iteraciones deben ser positivos.")
        if not 0 < nivel_minimo < 1:
            raise ValueError("El nivel mínimo de ventilación debe estar en (0, 1).")
        self.A, self.B, self.C = coef_ventilador, coef_calor, coef_acople
        self.vecinos = list(vecinos)
        self.alfa = tasa_aprendizaje
        self.tolerancia = tolerancia
        self.max_iteraciones = max_iteraciones
        self.nivel_minimo = nivel_minimo

    def energia(self, niveles: Sequence[float], cargas: Sequence[float]) -> float:
        u, carga = np.asarray(niveles, float), np.asarray(cargas, float)
        total = np.sum(self.A * u**3 + self.B * carga**2 / u)
        for i, j in self.vecinos:
            total += self.C * (u[i] - u[j]) ** 2
        return float(total)

    def gradiente(self, niveles: Sequence[float], cargas: Sequence[float]) -> np.ndarray:
        u, carga = np.asarray(niveles, float), np.asarray(cargas, float)
        grad = 3 * self.A * u**2 - self.B * carga**2 / u**2
        for i, j in self.vecinos:
            grad[i] += 2 * self.C * (u[i] - u[j])
            grad[j] += 2 * self.C * (u[j] - u[i])
        return grad

    def optimizar(
        self,
        cargas: Sequence[float],
        niveles_iniciales: Sequence[float] | None = None,
        minimos: Sequence[float] | None = None,
    ) -> ResultadoGradiente:
        """Minimiza E(u). `minimos` permite forzar ventilación (p. ej. emergencias)."""
        carga = np.asarray(cargas, float)
        if carga.ndim != 1 or carga.size == 0:
            raise ValueError("Se requiere al menos una carga.")
        if np.any(carga < 0) or np.any(carga > 1):
            raise ValueError("Las cargas deben estar entre 0 y 1.")

        limite_inferior = np.full(carga.size, self.nivel_minimo)
        if minimos is not None:
            limite_inferior = np.clip(np.asarray(minimos, float), self.nivel_minimo, 1.0)
        u = np.full(carga.size, 0.5) if niveles_iniciales is None else np.asarray(niveles_iniciales, float)
        u = np.clip(u, limite_inferior, 1.0)

        historial = [self.energia(u, carga)]
        trayectoria = [tuple(float(v) for v in u)]
        convergio = False
        iteracion = 0
        for iteracion in range(1, self.max_iteraciones + 1):
            nuevo = np.clip(u - self.alfa * self.gradiente(u, carga), limite_inferior, 1.0)
            paso = float(np.linalg.norm(nuevo - u))
            u = nuevo
            historial.append(self.energia(u, carga))
            trayectoria.append(tuple(float(v) for v in u))
            if paso < self.tolerancia:
                convergio = True
                break

        return ResultadoGradiente(
            niveles=tuple(float(v) for v in u),
            energia=historial[-1],
            iteraciones=iteracion,
            convergio=convergio,
            # Gradiente proyectado: en un límite activo solo cuenta la componente factible.
            norma_gradiente=float(
                np.linalg.norm(u - np.clip(u - self.gradiente(u, carga), limite_inferior, 1.0))
            ),
            historial_energia=tuple(historial),
            historial_niveles=tuple(trayectoria),
        )
