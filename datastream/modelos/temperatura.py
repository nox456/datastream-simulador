"""Módulo 2 · Análisis térmico con un modelo no lineal de una variable.

T(x) relaciona la carga del servidor x (% de CPU) con su temperatura (°C).
Con diferenciación se obtienen los puntos notables del modelo:

    1. Puntos estacionarios: soluciones de T'(x) = 0.
    2. Criterio de la segunda derivada en cada x0 estacionario:
           T''(x0) < 0  → máximo local
           T''(x0) > 0  → mínimo local
    3. Puntos de inflexión: T''(x) = 0 con cambio de concavidad.
    4. Punto crítico de emergencia: menor carga x_e con T(x_e) = T_emergencia
       y T creciente. Desde ahí el servidor requiere enfriamiento de emergencia.

Con el modelo por defecto la curva sube hasta un máximo local (el ventilador
interno trabaja a bajas RPM), baja hasta un mínimo local (el controlador
térmico entra en su segunda etapa) y desde ese mínimo crece de forma acelerada:
el mínimo local marca el fin de la zona térmicamente estable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property

import sympy as sp

x = sp.Symbol("x", real=True)

# Solo se aceptan estas funciones en la expresión (validación y seguridad:
# sympy evalúa el texto, por lo que no se permite ningún otro identificador).
FUNCIONES_PERMITIDAS = {"exp", "log", "sqrt", "sin", "cos", "tanh"}
_CARACTERES_VALIDOS = re.compile(r"^[0-9x+\-*/^().,\s a-z]+$")
_TOLERANCIA = 1e-9


@dataclass(frozen=True)
class PuntoNotable:
    carga: float  # % de CPU
    temperatura: float  # °C
    tipo: str


@dataclass(frozen=True)
class AnalisisTermico:
    funcion: str
    primera_derivada: str
    segunda_derivada: str
    puntos_estacionarios: tuple[PuntoNotable, ...]
    puntos_inflexion: tuple[PuntoNotable, ...]
    temperatura_emergencia: float
    carga_emergencia: float | None  # None si nunca se alcanza en el dominio
    zona_estable_hasta: float | None  # carga del último mínimo local
    maximo_absoluto: PuntoNotable


class ModeloTermico:
    """Modelo carga → temperatura y su análisis con derivadas."""

    def __init__(
        self,
        expresion: str,
        dominio: tuple[float, float] = (0.0, 100.0),
        temperatura_emergencia: float = 75.0,
    ):
        minimo, maximo = dominio
        if not minimo < maximo:
            raise ValueError("El dominio de carga debe cumplir mínimo < máximo.")
        self.expresion = self.interpretar(expresion)
        self.dominio = (float(minimo), float(maximo))
        self.temperatura_emergencia = float(temperatura_emergencia)
        self._evaluar = sp.lambdify(x, self.expresion, "math")
        self._derivada = sp.diff(self.expresion, x)

    @staticmethod
    def interpretar(texto: str) -> sp.Expr:
        """Convierte el texto en una expresión de sympy, validándolo."""
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("La función de temperatura debe ser un texto no vacío.")
        if not _CARACTERES_VALIDOS.match(texto):
            raise ValueError("La función contiene caracteres no permitidos.")
        sin_notacion_cientifica = re.sub(r"\d*\.?\d+e[+-]?\d+", "0", texto)  # 1e-3 no es identificador
        identificadores = set(re.findall(r"[a-z]+", sin_notacion_cientifica))
        desconocidos = identificadores - FUNCIONES_PERMITIDAS - {"x"}
        if desconocidos:
            raise ValueError(f"Identificadores no permitidos en la función: {sorted(desconocidos)}")
        try:
            expresion = sp.sympify(texto.replace("^", "**"), locals={"x": x})
        except (sp.SympifyError, SyntaxError, TypeError) as error:
            raise ValueError(f"No se pudo interpretar la función: {error}") from error
        if expresion.free_symbols != {x}:
            raise ValueError("La función debe depender únicamente de la variable x (carga).")
        return expresion

    # ------------------------------------------------------------------ #
    # Evaluación (se usa en cada paso de la simulación)
    # ------------------------------------------------------------------ #
    def temperatura(self, carga: float) -> float:
        minimo, maximo = self.dominio
        return float(self._evaluar(min(max(carga, minimo), maximo)))

    def requiere_emergencia(self, carga: float) -> bool:
        return self.temperatura(carga) >= self.temperatura_emergencia

    # ------------------------------------------------------------------ #
    # Análisis con derivadas
    # ------------------------------------------------------------------ #
    @cached_property
    def analisis(self) -> AnalisisTermico:
        t1 = self._derivada
        t2 = sp.diff(t1, x)

        estacionarios = []
        for raiz in self._raices(t1):
            curvatura = float(t2.subs(x, raiz))
            if curvatura < -_TOLERANCIA:
                tipo = "máximo local"
            elif curvatura > _TOLERANCIA:
                tipo = "mínimo local"
            else:
                tipo = "punto silla (inflexión horizontal)"
            estacionarios.append(PuntoNotable(raiz, self.temperatura(raiz), tipo))

        inflexiones = [
            PuntoNotable(raiz, self.temperatura(raiz), "punto de inflexión")
            for raiz in self._raices(t2)
            if self._cambia_signo(t2, raiz)
        ]

        minimos = [p.carga for p in estacionarios if p.tipo == "mínimo local"]
        return AnalisisTermico(
            funcion=str(self.expresion),
            primera_derivada=str(sp.simplify(t1)),
            segunda_derivada=str(sp.simplify(t2)),
            puntos_estacionarios=tuple(estacionarios),
            puntos_inflexion=tuple(inflexiones),
            temperatura_emergencia=self.temperatura_emergencia,
            carga_emergencia=self._carga_emergencia(),
            zona_estable_hasta=max(minimos) if minimos else None,
            maximo_absoluto=self._maximo_absoluto(estacionarios),
        )

    def analizar(self) -> AnalisisTermico:
        return self.analisis

    def _carga_emergencia(self) -> float | None:
        """Menor carga donde T(x) alcanza T_emergencia subiendo."""
        minimo, _ = self.dominio
        if self.temperatura(minimo) >= self.temperatura_emergencia:
            return minimo
        cruces = [
            raiz
            for raiz in self._raices(self.expresion - self.temperatura_emergencia)
            if float(self._derivada.subs(x, raiz)) > 0
        ]
        return min(cruces) if cruces else None

    def _maximo_absoluto(self, estacionarios: list[PuntoNotable]) -> PuntoNotable:
        """El máximo absoluto está en un punto estacionario o en un extremo."""
        candidatos = [p.carga for p in estacionarios] + list(self.dominio)
        mejor = max(candidatos, key=self.temperatura)
        return PuntoNotable(mejor, self.temperatura(mejor), "máximo absoluto en el dominio")

    def _raices(self, expresion: sp.Expr) -> list[float]:
        """Raíces reales de `expresion` dentro del dominio.

        Primero se intenta la solución analítica (solveset); si sympy no puede
        resolverla de forma cerrada se usa un método numérico (bisección).
        """
        minimo, maximo = self.dominio
        try:
            solucion = sp.solveset(sp.Eq(expresion, 0), x, domain=sp.Interval(minimo, maximo))
            if isinstance(solucion, sp.FiniteSet):
                return sorted(float(sp.re(r.evalf())) for r in solucion)
            if solucion is sp.S.EmptySet:
                return []
        except (NotImplementedError, TypeError, ValueError):
            pass
        return self._raices_numericas(expresion)

    def _raices_numericas(self, expresion: sp.Expr, muestras: int = 2000) -> list[float]:
        funcion = sp.lambdify(x, expresion, "math")
        minimo, maximo = self.dominio
        paso = (maximo - minimo) / muestras
        raices = []
        a, fa = minimo, funcion(minimo)
        for i in range(1, muestras + 1):
            b = minimo + i * paso
            fb = funcion(b)
            if fa == 0:
                raices.append(a)
            elif fa * fb < 0:
                for _ in range(60):  # bisección
                    medio = (a + b) / 2
                    if fa * funcion(medio) <= 0:
                        b = medio
                    else:
                        a, fa = medio, funcion(medio)
                raices.append((a + b) / 2)
            a, fa = b, fb
        return [r for i, r in enumerate(raices) if i == 0 or not math.isclose(r, raices[i - 1])]

    def _cambia_signo(self, expresion: sp.Expr, punto: float, delta: float = 1e-3) -> bool:
        antes = float(expresion.subs(x, punto - delta))
        despues = float(expresion.subs(x, punto + delta))
        return antes * despues < 0
