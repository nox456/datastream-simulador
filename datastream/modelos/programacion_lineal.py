"""Programación lineal · Método Simplex de dos fases.

Problema:
    max (o min) Z = c_1·x_1 + … + c_n·x_n
    s.a.  a_i1·x_1 + … + a_in·x_n  (<=, >= o =)  b_i,   i = 1, …, m
          x_j >= 0

Forma estándar: si b_i < 0 la restricción se multiplica por −1 (y cambia su sentido);
cada restricción <= recibe una holgura s_i, cada >= un exceso e_i (coeficiente −1) y
una artificial a_i, y cada = una artificial a_i. Minimizar Z equivale a maximizar −Z.

Fase 1 (solo si hay artificiales): max W = −Σ a_i. Si W* < 0 alguna artificial no
puede anularse y el problema es infactible. Fase 2: desde la base factible que dejó
la fase 1, sin las columnas artificiales, se optimiza Z.

En cada tabla la fila objetivo guarda z_j − c_j:
    - entra la variable con el z_j − c_j más negativo;
    - sale la variable básica con la menor razón b_i / a_ik entre los a_ik > 0;
    - si ningún a_ik es positivo, Z crece sin límite: la solución es no acotada;
    - si todos los z_j − c_j son >= 0 la tabla es óptima; una variable no básica con
      z_j − c_j = 0 que al entrar cambia la solución indica soluciones múltiples.

Si una base se repite (ciclaje por degeneración) se pasa a la regla de Bland.
Los precios sombra y_i = ∂Z/∂b_i salen de la base óptima: yᵀ = c_Bᵀ·B⁻¹.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

SENTIDOS = ("<=", ">=", "=")
OBJETIVOS = ("max", "min")
ESTADOS = ("optimo", "no_acotado", "infactible")
TOLERANCIA = 1e-9
_INVERSO = {"<=": ">=", ">=": "<=", "=": "="}


@dataclass(frozen=True)
class Restriccion:
    """a_i·x (sentido) b_i."""

    coeficientes: tuple[float, ...]
    sentido: str
    lado_derecho: float


@dataclass(frozen=True)
class ProblemaLineal:
    objetivo: tuple[float, ...]
    restricciones: tuple[Restriccion, ...]
    sentido: str = "max"
    variables: tuple[str, ...] = ()

    @property
    def nombres(self) -> tuple[str, ...]:
        return tuple(self.variables) or tuple(f"x{j}" for j in range(1, len(self.objetivo) + 1))


@dataclass(frozen=True)
class TablaSimplex:
    """Una tabla del Simplex tal como se ve antes de pivotear.

    filas: coeficientes de cada restricción en la forma estándar y, al final, b_i.
    fila_objetivo: z_j − c_j de cada columna y, al final, el valor actual del objetivo
    de la fase (W en la fase 1; Z, o −Z si se minimiza, en la fase 2).
    entrante, saliente y razones describen el pivote que se hace a partir de esta tabla;
    entrante es None en la tabla óptima y saliente es None si la solución es no acotada.
    """

    fase: int
    iteracion: int
    columnas: tuple[str, ...]
    base: tuple[str, ...]
    filas: tuple[tuple[float, ...], ...]
    fila_objetivo: tuple[float, ...]
    etiqueta_objetivo: str
    entrante: str | None = None
    saliente: str | None = None
    razones: tuple[float | None, ...] = ()


@dataclass(frozen=True)
class ResultadoSimplex:
    """Resultado del Simplex.

    estado: "optimo", "no_acotado" o "infactible". Los valores, el objetivo, los precios
    sombra y las holguras solo se llenan cuando el estado es "optimo".
    precios_sombra[i] = ∂Z/∂b_i: cambio de Z por cada unidad adicional de b_i.
    holguras[i] = b_i − a_i·x en <=, a_i·x − b_i en >= y 0 en =.
    """

    estado: str
    variables: tuple[str, ...]
    tablas: tuple[TablaSimplex, ...]
    valores: tuple[float, ...] = ()
    valor_objetivo: float | None = None
    precios_sombra: tuple[float, ...] = ()
    holguras: tuple[float, ...] = ()
    soluciones_multiples: bool = False
    solucion_alternativa: tuple[float, ...] | None = None
    pivote_alternativo: tuple[str, str] | None = None
    tabla_alternativa: TablaSimplex | None = None
    variable_no_acotada: str | None = None
    degenerada: bool = False
    restricciones_redundantes: tuple[int, ...] = ()

    @property
    def es_optimo(self) -> bool:
        return self.estado == "optimo"

    @property
    def usa_dos_fases(self) -> bool:
        return any(t.fase == 1 for t in self.tablas)


def validar_problema(problema: ProblemaLineal) -> None:
    if problema.sentido not in OBJETIVOS:
        raise ValueError("El objetivo debe ser 'max' o 'min'.")
    n = len(problema.objetivo)
    if n == 0:
        raise ValueError("La función objetivo necesita al menos una variable.")
    if not problema.restricciones:
        raise ValueError("Se requiere al menos una restricción.")
    if problema.variables and len(problema.variables) != n:
        raise ValueError(f"Se dieron {len(problema.variables)} nombres para {n} variables.")
    numeros = list(problema.objetivo)
    for i, r in enumerate(problema.restricciones, start=1):
        if r.sentido not in SENTIDOS:
            raise ValueError(f"La restricción {i} tiene un sentido inválido '{r.sentido}' (use <=, >= o =).")
        if len(r.coeficientes) != n:
            raise ValueError(f"La restricción {i} tiene {len(r.coeficientes)} coeficientes y la función objetivo {n}.")
        numeros += [*r.coeficientes, r.lado_derecho]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in numeros):
        raise ValueError("Todos los coeficientes y lados derechos deben ser números finitos.")


class ResolvedorSimplex:
    """Método Simplex de dos fases que guarda cada tabla del proceso."""

    def __init__(self, max_iteraciones: int = 200) -> None:
        if max_iteraciones < 1:
            raise ValueError("Se necesita al menos una iteración.")
        self.max_iteraciones = max_iteraciones

    def resolver(self, problema: ProblemaLineal) -> ResultadoSimplex:
        validar_problema(problema)
        n, m = len(problema.objetivo), len(problema.restricciones)
        signo = 1.0 if problema.sentido == "max" else -1.0
        nombres = problema.nombres

        normalizadas, invertidas = [], []
        for r in problema.restricciones:
            coeficientes, b, sentido = [float(a) for a in r.coeficientes], float(r.lado_derecho), r.sentido
            invertidas.append(b < 0)
            if b < 0:
                coeficientes, b, sentido = [-a for a in coeficientes], -b, _INVERSO[sentido]
            normalizadas.append((coeficientes, sentido, b))

        extras = []
        for i, (_, sentido, _) in enumerate(normalizadas):
            if sentido == "<=":
                extras.append((f"s{i + 1}", i, 1.0, False))
                continue
            if sentido == ">=":
                extras.append((f"e{i + 1}", i, -1.0, False))
            extras.append((f"a{i + 1}", i, 1.0, True))

        columnas = list(nombres) + [e[0] for e in extras]
        matriz = [coeficientes + [0.0] * len(extras) + [b] for coeficientes, _, b in normalizadas]
        base: list[int] = [0] * m
        artificiales = set()
        for j, (_, i, valor, artificial) in enumerate(extras, start=n):
            matriz[i][j] = valor
            if artificial:
                artificiales.add(j)
            if valor > 0:
                base[i] = j
        inicial = [fila[:-1] for fila in matriz]
        costos = [signo * float(c) for c in problema.objetivo] + [0.0] * len(extras)
        origen = list(range(m))
        tablas: list[TablaSimplex] = []
        redundantes: list[int] = []

        if artificiales:
            fila_w = [1.0 if j in artificiales else 0.0 for j in range(len(columnas))] + [0.0]
            for fila, b in zip(matriz, base):
                if b in artificiales:
                    fila_w = [w - v for w, v in zip(fila_w, fila)]
            self._iterar(1, matriz, fila_w, base, columnas, range(len(columnas)), tablas, "W")
            if fila_w[-1] < -TOLERANCIA * max(1.0, sum(b for _, _, b in normalizadas)):
                return ResultadoSimplex("infactible", nombres, tuple(tablas))

            r = 0
            while r < len(base):
                if base[r] in artificiales:
                    k = next((j for j in range(len(columnas)) if j not in artificiales and abs(matriz[r][j]) > TOLERANCIA), None)
                    if k is None:
                        redundantes.append(origen[r])
                        del matriz[r], base[r], origen[r]
                        continue
                    self._pivotear(matriz, fila_w, r, k)
                    base[r] = k
                r += 1

        conservadas = [j for j in range(len(columnas)) if j not in artificiales]
        posicion = {j: p for p, j in enumerate(conservadas)}
        matriz = [[fila[j] for j in conservadas] + [fila[-1]] for fila in matriz]
        base = [posicion[b] for b in base]
        columnas_fase2 = [columnas[j] for j in conservadas]
        costos_fase2 = [costos[j] for j in conservadas]

        fila_z = [-c for c in costos_fase2] + [0.0]
        for fila, b in zip(matriz, base):
            if costos_fase2[b]:
                fila_z = [z + costos_fase2[b] * v for z, v in zip(fila_z, fila)]
        etiqueta = "Z" if signo > 0 else "-Z"
        no_acotada = self._iterar(2, matriz, fila_z, base, columnas_fase2, range(len(columnas_fase2)), tablas, etiqueta)
        if no_acotada is not None:
            return ResultadoSimplex("no_acotado", nombres, tuple(tablas), variable_no_acotada=no_acotada)

        valores = self._valores(matriz, base, n)
        matriz_base = np.array([[inicial[origen[i]][conservadas[b]] for b in base] for i in range(len(base))])
        duales = np.linalg.solve(matriz_base.T, np.array([costos_fase2[b] for b in base])) if base else []
        precios = [0.0] * m
        for i, y in zip(origen, duales):
            precios[i] = _limpio(signo * (-1.0 if invertidas[i] else 1.0) * float(y))

        holguras = []
        for r in problema.restricciones:
            actividad = sum(a * v for a, v in zip(r.coeficientes, valores))
            holguras.append(_limpio(actividad - r.lado_derecho if r.sentido == ">=" else r.lado_derecho - actividad))

        multiples, alternativa, pivote, tabla_alternativa = self._alternativa(
            matriz, fila_z, base, columnas_fase2, n, etiqueta, len(tablas)
        )
        return ResultadoSimplex(
            "optimo",
            nombres,
            tuple(tablas),
            valores=valores,
            valor_objetivo=_limpio(signo * fila_z[-1]),
            precios_sombra=tuple(precios),
            holguras=tuple(holguras),
            soluciones_multiples=multiples,
            solucion_alternativa=alternativa,
            pivote_alternativo=pivote,
            tabla_alternativa=tabla_alternativa,
            degenerada=any(abs(fila[-1]) <= TOLERANCIA for fila in matriz),
            restricciones_redundantes=tuple(sorted(redundantes)),
        )

    def _iterar(
        self,
        fase: int,
        matriz: list[list[float]],
        fila_objetivo: list[float],
        base: list[int],
        columnas: Sequence[str],
        permitidas: Sequence[int],
        tablas: list[TablaSimplex],
        etiqueta: str,
    ) -> str | None:
        """Pivotea hasta la tabla óptima. Devuelve la variable que crece sin límite, o None."""
        vistas = {tuple(sorted(base))}
        bland = False
        for iteracion in range(self.max_iteraciones + 1):
            entrante = self._entrante(fila_objetivo, permitidas, bland)
            if entrante is None:
                tablas.append(self._foto(fase, iteracion, columnas, base, matriz, fila_objetivo, etiqueta))
                return None
            razones = tuple(fila[-1] / fila[entrante] if fila[entrante] > TOLERANCIA else None for fila in matriz)
            candidatas = [i for i, razon in enumerate(razones) if razon is not None]
            if not candidatas:
                tablas.append(self._foto(fase, iteracion, columnas, base, matriz, fila_objetivo, etiqueta,
                                         columnas[entrante], None, razones))
                return columnas[entrante]
            minima = min(razones[i] for i in candidatas)
            r = min((i for i in candidatas if razones[i] <= minima + TOLERANCIA), key=lambda i: base[i])
            tablas.append(self._foto(fase, iteracion, columnas, base, matriz, fila_objetivo, etiqueta,
                                     columnas[entrante], columnas[base[r]], razones))
            self._pivotear(matriz, fila_objetivo, r, entrante)
            base[r] = entrante
            clave = tuple(sorted(base))
            bland = bland or clave in vistas
            vistas.add(clave)
        raise RuntimeError(f"El Simplex no terminó en {self.max_iteraciones} iteraciones.")

    @staticmethod
    def _entrante(fila_objetivo: Sequence[float], permitidas: Sequence[int], bland: bool) -> int | None:
        negativas = [j for j in permitidas if fila_objetivo[j] < -TOLERANCIA]
        if not negativas:
            return None
        return negativas[0] if bland else min(negativas, key=lambda j: (fila_objetivo[j], j))

    @staticmethod
    def _pivotear(matriz: list[list[float]], fila_objetivo: list[float], r: int, k: int) -> None:
        pivote = matriz[r][k]
        matriz[r] = [_limpio(v / pivote) for v in matriz[r]]
        for i, fila in enumerate(matriz):
            if i != r and fila[k] != 0.0:
                factor = fila[k]
                matriz[i] = [_limpio(a - factor * p) for a, p in zip(fila, matriz[r])]
        factor = fila_objetivo[k]
        fila_objetivo[:] = [_limpio(a - factor * p) for a, p in zip(fila_objetivo, matriz[r])]

    @staticmethod
    def _valores(matriz: Sequence[Sequence[float]], base: Sequence[int], n: int) -> tuple[float, ...]:
        valores = [0.0] * n
        for fila, b in zip(matriz, base):
            if b < n:
                valores[b] = _limpio(fila[-1])
        return tuple(valores)

    def _alternativa(
        self,
        matriz: list[list[float]],
        fila_z: list[float],
        base: list[int],
        columnas: Sequence[str],
        n: int,
        etiqueta: str,
        iteracion: int,
    ) -> tuple[bool, tuple[float, ...] | None, tuple[str, str] | None, TablaSimplex | None]:
        """Busca otra solución óptima haciendo entrar una no básica con z_j − c_j = 0."""
        actuales = self._valores(matriz, base, n)
        for k in range(len(columnas)):
            if k in base or abs(fila_z[k]) > TOLERANCIA:
                continue
            positivas = [i for i, fila in enumerate(matriz) if fila[k] > TOLERANCIA]
            if not positivas:
                if k < n or any(fila[k] < -TOLERANCIA for fila, b in zip(matriz, base) if b < n):
                    return True, None, None, None
                continue
            minima = min(matriz[i][-1] / matriz[i][k] for i in positivas)
            if minima <= TOLERANCIA:
                continue
            r = min((i for i in positivas if matriz[i][-1] / matriz[i][k] <= minima + TOLERANCIA), key=lambda i: base[i])
            copia, objetivo, nueva_base = [f[:] for f in matriz], fila_z[:], base[:]
            self._pivotear(copia, objetivo, r, k)
            nueva_base[r] = k
            valores = self._valores(copia, nueva_base, n)
            if any(abs(a - b) > TOLERANCIA for a, b in zip(valores, actuales)):
                tabla = self._foto(2, iteracion, columnas, nueva_base, copia, objetivo, etiqueta)
                return True, valores, (columnas[k], columnas[base[r]]), tabla
        return False, None, None, None

    @staticmethod
    def _foto(
        fase: int,
        iteracion: int,
        columnas: Sequence[str],
        base: Sequence[int],
        matriz: Sequence[Sequence[float]],
        fila_objetivo: Sequence[float],
        etiqueta: str,
        entrante: str | None = None,
        saliente: str | None = None,
        razones: tuple[float | None, ...] = (),
    ) -> TablaSimplex:
        return TablaSimplex(
            fase=fase,
            iteracion=iteracion,
            columnas=tuple(columnas),
            base=tuple(columnas[b] for b in base),
            filas=tuple(tuple(fila) for fila in matriz),
            fila_objetivo=tuple(fila_objetivo),
            etiqueta_objetivo=etiqueta,
            entrante=entrante,
            saliente=saliente,
            razones=razones,
        )


def resolver_con_scipy(problema: ProblemaLineal) -> tuple[str, tuple[float, ...], float | None]:
    """Resuelve el mismo problema con scipy.optimize.linprog (HiGHS) para verificar el Simplex propio."""
    from scipy.optimize import linprog

    validar_problema(problema)
    signo = 1.0 if problema.sentido == "max" else -1.0
    desigualdades, cotas, igualdades, valores_igualdad = [], [], [], []
    for r in problema.restricciones:
        if r.sentido == "=":
            igualdades.append(list(r.coeficientes))
            valores_igualdad.append(r.lado_derecho)
        else:
            factor = 1.0 if r.sentido == "<=" else -1.0
            desigualdades.append([factor * a for a in r.coeficientes])
            cotas.append(factor * r.lado_derecho)
    solucion = linprog(
        [-signo * c for c in problema.objetivo],
        A_ub=desigualdades or None,
        b_ub=cotas or None,
        A_eq=igualdades or None,
        b_eq=valores_igualdad or None,
        bounds=[(0, None)] * len(problema.objetivo),
        method="highs",
    )
    estado = {0: "optimo", 2: "infactible", 3: "no_acotado"}.get(solucion.status, "error")
    if estado != "optimo":
        return estado, (), None
    return estado, tuple(float(v) for v in solucion.x), -signo * float(solucion.fun)


def _limpio(valor: float) -> float:
    return 0.0 if abs(valor) < 1e-12 else valor
