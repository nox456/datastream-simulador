"""Referencia teórica de colas M/M/c (fórmula de Erlang C).

Sirve para contrastar los resultados de la simulación: llegadas de Poisson con
tasa λ, tiempos de servicio exponenciales con tasa μ y c servidores en paralelo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricasMMc:
    tasa_llegada: float  # λ
    tasa_servicio: float  # μ (por servidor)
    servidores: int  # c
    utilizacion: float  # ρ = λ / (c·μ)
    estable: bool
    prob_espera: float | None = None  # Erlang C: P(esperar)
    lq: float | None = None  # clientes promedio en cola
    wq: float | None = None  # espera promedio en cola
    w: float | None = None  # tiempo promedio en el sistema
    l: float | None = None  # clientes promedio en el sistema


def calcular_mmc(tasa_llegada: float, tasa_servicio: float, servidores: int) -> MetricasMMc:
    if tasa_llegada <= 0 or tasa_servicio <= 0 or servidores < 1:
        raise ValueError("λ y μ deben ser positivos y c >= 1.")
    c = servidores
    a = tasa_llegada / tasa_servicio  # carga ofrecida (Erlangs)
    rho = a / c
    if rho >= 1:
        return MetricasMMc(tasa_llegada, tasa_servicio, c, rho, estable=False)

    # P0 = [ Σ_{k=0}^{c-1} a^k/k!  +  a^c / (c!·(1-ρ)) ]^-1   (en forma iterativa estable)
    termino, suma = 1.0, 1.0
    for k in range(1, c):
        termino *= a / k
        suma += termino
    termino_c = termino * a / c  # a^c / c!
    cola = termino_c / (1 - rho)
    p0 = 1 / (suma + cola)
    prob_espera = cola * p0
    lq = prob_espera * rho / (1 - rho)
    wq = lq / tasa_llegada
    w = wq + 1 / tasa_servicio
    return MetricasMMc(tasa_llegada, tasa_servicio, c, rho, True, prob_espera, lq, wq, w, tasa_llegada * w)


def ley_de_little(numero_promedio: float, tasa_efectiva: float, tiempo_promedio: float) -> dict[str, float]:
    """Compara L observado con λ·W (deben ser parecidos en régimen estable)."""
    estimado = tasa_efectiva * tiempo_promedio
    error = abs(numero_promedio - estimado) / estimado if estimado else math.nan
    return {"L_observado": numero_promedio, "lambda_por_W": estimado, "error_relativo": error}
