"""Pruebas del Módulo 1: ruta más corta (DP por etapas) y volumen-carga (mochila)."""

import itertools
import random

import pytest

from datastream.modelos.mochila import Articulo, ModeloVolumenCarga
from datastream.modelos.ruta_mas_corta import RutaMasCortaDP

ETAPAS = [["A"], ["B", "C"], ["D", "E"], ["F"]]
ARCOS = {
    ("A", "B"): 2.0, ("A", "C"): 4.0,
    ("B", "D"): 7.0, ("B", "E"): 3.0,
    ("C", "D"): 1.0, ("C", "E"): 4.0,
    ("D", "F"): 1.0, ("E", "F"): 5.0,
}


def test_ruta_optima_por_etapas():
    resultado = RutaMasCortaDP(ETAPAS).resolver(ARCOS)
    # A→C→D→F = 4 + 1 + 1 = 6 es mejor que A→B→E→F = 10 y A→B→D→F = 10
    assert resultado.ruta_hacia("F") == ["A", "C", "D", "F"]
    assert resultado.latencia_hacia("F") == pytest.approx(6.0)
    assert resultado.distancias["E"] == pytest.approx(5.0)  # A→B→E


def test_tabla_contiene_todos_los_estados():
    resultado = RutaMasCortaDP(ETAPAS).resolver(ARCOS)
    assert [fila.nodo for fila in resultado.tabla] == ["B", "C", "D", "E", "F"]
    fila_d = next(f for f in resultado.tabla if f.nodo == "D")
    assert {c.desde for c in fila_d.candidatos} == {"B", "C"}
    assert fila_d.decision == "C"


def test_nodo_caido_cambia_la_ruta():
    sin_c = {k: v for k, v in ARCOS.items() if "C" not in k}
    resultado = RutaMasCortaDP(ETAPAS).resolver(sin_c)
    assert resultado.ruta_hacia("F") == ["A", "B", "D", "F"]
    assert not resultado.es_alcanzable("C")
    assert resultado.ruta_hacia("C") is None


def test_la_congestion_cambia_la_ruta():
    congestionado = dict(ARCOS)
    congestionado[("C", "D")] = 9.0
    assert RutaMasCortaDP(ETAPAS).resolver(congestionado).ruta_hacia("F") == ["A", "B", "D", "F"]


def test_validaciones_ruta():
    with pytest.raises(ValueError):
        RutaMasCortaDP([["A", "B"], ["C"]])  # dos orígenes
    with pytest.raises(ValueError):
        RutaMasCortaDP([["A"], ["A"]])  # nodo repetido
    with pytest.raises(ValueError):
        RutaMasCortaDP(ETAPAS).resolver({("A", "B"): -1.0})


def _fuerza_bruta(articulos, capacidad):
    mejor = 0.0
    for r in range(len(articulos) + 1):
        for combinacion in itertools.combinations(articulos, r):
            if sum(a.peso for a in combinacion) <= capacidad:
                mejor = max(mejor, sum(a.valor for a in combinacion))
    return mejor


def test_mochila_ejemplo_clasico():
    articulos = [Articulo("a", 2, 3), Articulo("b", 3, 4), Articulo("c", 4, 5), Articulo("d", 5, 6)]
    resultado = ModeloVolumenCarga().resolver(articulos, 5)
    assert resultado.valor_total == 7
    assert resultado.seleccionados == (0, 1)
    assert resultado.peso_total == 5


@pytest.mark.parametrize("semilla", range(15))
def test_mochila_coincide_con_fuerza_bruta(semilla):
    azar = random.Random(semilla)
    articulos = [Articulo(f"T{i}", azar.randint(1, 8), azar.uniform(1, 20)) for i in range(azar.randint(1, 9))]
    capacidad = azar.randint(0, 20)
    resultado = ModeloVolumenCarga().resolver(articulos, capacidad)
    assert resultado.valor_total == pytest.approx(_fuerza_bruta(articulos, capacidad))
    assert resultado.peso_total <= capacidad
    assert resultado.valor_total == pytest.approx(sum(articulos[i].valor for i in resultado.seleccionados))


def test_mochila_tabla_y_casos_borde():
    modelo = ModeloVolumenCarga()
    assert modelo.resolver([], 10).valor_total == 0
    assert modelo.resolver([Articulo("x", 5, 9)], 4).seleccionados == ()
    resultado = modelo.resolver([Articulo("x", 2, 3)], 3, guardar_tabla=True)
    assert resultado.tabla == ((0, 0, 0, 0), (0, 0, 3, 3))


def test_mochila_validaciones():
    with pytest.raises(ValueError):
        ModeloVolumenCarga().resolver([Articulo("x", 0, 1)], 5)
    with pytest.raises(ValueError):
        ModeloVolumenCarga().resolver([Articulo("x", 1, -1)], 5)
    with pytest.raises(ValueError):
        ModeloVolumenCarga().resolver([], -1)
