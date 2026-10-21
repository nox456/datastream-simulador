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


# ---------------------------------------------------------------- recursión hacia atrás
def test_recursion_hacia_atras_en_la_red_de_prueba():
    resultado = RutaMasCortaDP(ETAPAS).resolver_hacia_atras(ARCOS, "F")
    assert resultado.ruta() == ["A", "C", "D", "F"] and resultado.latencia == pytest.approx(6.0)
    assert resultado.valores["C"] == pytest.approx(2.0)  # C→D→F
    assert [f.etapa for f in resultado.tabla] == [3, 3, 2, 2, 1]  # se calcula desde la última etapa


@pytest.mark.parametrize("semilla", range(10))
def test_ambas_recursiones_coinciden(semilla):
    azar = random.Random(semilla)
    etapas = [["O"], ["A", "B", "C"], ["D", "E"], ["F", "G", "H"], ["Z"]]
    arcos = {
        (x, y): azar.uniform(0.1, 10)
        for k in range(len(etapas) - 1)
        for x in etapas[k]
        for y in etapas[k + 1]
        if azar.random() < 0.8
    }
    dp = RutaMasCortaDP(etapas)
    adelante, atras = dp.resolver(arcos), dp.resolver_hacia_atras(arcos, "Z")
    assert atras.latencia == pytest.approx(adelante.latencia_hacia("Z"))
    if atras.ruta():
        assert sum(arcos[t] for t in zip(atras.ruta(), atras.ruta()[1:])) == pytest.approx(atras.latencia)


def test_destino_inexistente_en_recursion_hacia_atras():
    with pytest.raises(ValueError):
        RutaMasCortaDP(ETAPAS).resolver_hacia_atras(ARCOS, "Z")


# ---------------------------------------------------------------- ejemplos resueltos en clase
from datastream.consola import ARCOS_CLASE, ARTICULOS_BARCO, CAPACIDAD_BARCO, ETAPAS_CLASE  # noqa: E402
from datastream.modelos.mochila import ModeloMochilaUnidades  # noqa: E402


def test_ejemplo_de_clase_hacia_adelante():
    resultado = RutaMasCortaDP(ETAPAS_CLASE).resolver(ARCOS_CLASE)
    assert resultado.ruta_hacia("7") == ["1", "4", "5", "7"]
    assert resultado.latencia_hacia("7") == 21
    assert (resultado.distancias["5"], resultado.distancias["6"]) == (12, 17)  # lámina 7


def test_ejemplo_de_clase_hacia_atras_reproduce_las_tablas():
    resultado = RutaMasCortaDP(ETAPAS_CLASE).resolver_hacia_atras(ARCOS_CLASE, "7")
    assert resultado.ruta() == ["1", "4", "5", "7"] and resultado.latencia == 21
    # Láminas 11 a 13: f_3(5)=9, f_3(6)=6, f_2(2)=21, f_2(3)=15, f_2(4)=16 y f_1(1)=21
    assert {n: resultado.valores[n] for n in "234567"} == {"2": 21, "3": 15, "4": 16, "5": 9, "6": 6, "7": 0}
    assert [f.etapa for f in resultado.tabla] == [3, 3, 2, 2, 2, 1]  # se calcula desde la última etapa
    fila_3 = next(f for f in resultado.tabla if f.nodo == "3")
    assert [(c.hacia, c.costo) for c in fila_3.candidatos] == [("5", 17), ("6", 15)] and fila_3.decision == "6"


def test_mochila_del_barco_de_clase():
    resultado = ModeloMochilaUnidades().resolver(ARTICULOS_BARCO, CAPACIDAD_BARCO)
    assert resultado.cantidades == (2, 0, 0) and resultado.valor_total == 62  # lámina 21
    etapa_3, etapa_2, etapa_1 = resultado.etapas
    assert [f.valor_optimo for f in etapa_3.filas] == [0, 14, 28, 42, 56]  # lámina 19
    assert [f.valor_optimo for f in etapa_2.filas] == [0, 14, 28, 47, 61]  # lámina 20
    assert [f.valor_optimo for f in etapa_1.filas] == [0, 14, 31, 47, 62]  # lámina 21
    assert etapa_1.filas[4].alternativas == (61, 59, 62)


@pytest.mark.parametrize("semilla", range(10))
def test_mochila_por_unidades_con_limite_1_es_la_mochila_0_1(semilla):
    azar = random.Random(semilla)
    articulos = [Articulo(f"T{i}", azar.randint(1, 8), azar.uniform(1, 20)) for i in range(azar.randint(1, 8))]
    capacidad = azar.randint(0, 20)
    unidades = ModeloMochilaUnidades().resolver(articulos, capacidad, limites=[1] * len(articulos))
    assert unidades.valor_total == pytest.approx(ModeloVolumenCarga().resolver(articulos, capacidad).valor_total)
    assert unidades.peso_total <= capacidad
