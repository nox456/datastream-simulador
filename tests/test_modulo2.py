"""Pruebas del Módulo 2: modelo térmico, método de gradiente y Lagrange."""

import math
import random

import numpy as np
import pytest

from datastream.modelos.colas import calcular_mmc
from datastream.modelos.presupuesto import AsignadorEnergia
from datastream.modelos.refrigeracion import OptimizadorRefrigeracion
from datastream.modelos.temperatura import ModeloTermico

FUNCION = "0.0004*x**3 - 0.054*x**2 + 2.16*x + 25"


# ---------------------------------------------------------------- térmico
def test_puntos_estacionarios_e_inflexion():
    analisis = ModeloTermico(FUNCION, (0, 100), 75).analizar()
    tipos = {p.tipo: p for p in analisis.puntos_estacionarios}
    assert tipos["máximo local"].carga == pytest.approx(30)
    assert tipos["máximo local"].temperatura == pytest.approx(52)
    assert tipos["mínimo local"].carga == pytest.approx(60)
    assert analisis.puntos_inflexion[0].carga == pytest.approx(45)
    assert analisis.zona_estable_hasta == pytest.approx(60)


def test_punto_critico_de_emergencia():
    modelo = ModeloTermico(FUNCION, (0, 100), 75)
    x_e = modelo.analizar().carga_emergencia
    assert modelo.temperatura(x_e) == pytest.approx(75, abs=1e-6)
    assert not modelo.requiere_emergencia(x_e - 1)
    assert modelo.requiere_emergencia(x_e + 1)


def test_umbral_inalcanzable_y_funcion_no_polinomica():
    assert ModeloTermico(FUNCION, (0, 100), 500).analizar().carga_emergencia is None
    logistica = ModeloTermico("25 + 60/(1 + exp(-0.1*(x - 70)))", (0, 100), 75).analizar()
    assert logistica.puntos_inflexion[0].carga == pytest.approx(70, abs=1e-3)
    assert logistica.puntos_estacionarios == ()


@pytest.mark.parametrize("expresion", ["", "25", "y**2 + x", "__import__('os')", "x + open(1)", "x +* 2"])
def test_funcion_invalida_se_rechaza(expresion):
    with pytest.raises(ValueError):
        ModeloTermico(expresion)


def test_notacion_cientifica_permitida():
    assert ModeloTermico("1e-3*x**2 + 25").temperatura(10) == pytest.approx(25.1)


# ---------------------------------------------------------------- gradiente
def _optimizador(acople=0.8):
    return OptimizadorRefrigeracion(4.0, 1.5, acople, [(0, 1), (1, 2), (2, 3)], 0.02, 1e-7, 2000, 0.1)


def test_gradiente_coincide_con_diferencias_finitas():
    opt, u, cargas, h = _optimizador(), np.array([0.3, 0.5, 0.7, 0.9]), [0.9, 0.4, 0.6, 0.1], 1e-6
    numerico = []
    for i in range(4):
        mas, menos = u.copy(), u.copy()
        mas[i] += h
        menos[i] -= h
        numerico.append((opt.energia(mas, cargas) - opt.energia(menos, cargas)) / (2 * h))
    assert opt.gradiente(u, cargas) == pytest.approx(numerico, rel=1e-5)


def test_optimo_sin_acople_es_el_analitico():
    # Con C = 0: 3A·u² = B·L²/u²  ⇒  u* = (B·L² / 3A)^(1/4)
    cargas = [0.9, 0.5, 0.3, 0.7]
    resultado = _optimizador(acople=0).optimizar(cargas)
    esperado = [(1.5 * carga**2 / 12) ** 0.25 for carga in cargas]
    assert resultado.convergio
    assert resultado.niveles == pytest.approx(esperado, abs=1e-4)


def test_gradiente_reduce_la_energia_y_respeta_limites():
    resultado = _optimizador().optimizar([1.0, 0.0, 0.5, 0.2], minimos=[0.1, 0.1, 1.0, 0.1])
    energias = resultado.historial_energia
    assert energias[-1] < energias[0]
    assert all(0.1 <= u <= 1.0 for u in resultado.niveles)
    assert resultado.niveles[2] == pytest.approx(1.0)  # ventilación forzada (emergencia)
    assert resultado.niveles[1] < resultado.niveles[0]  # sin carga necesita menos aire


def test_gradiente_validaciones():
    with pytest.raises(ValueError):
        _optimizador().optimizar([1.5, 0.2, 0.2, 0.2])
    with pytest.raises(ValueError):
        OptimizadorRefrigeracion(-1, 1, 1, [])


# ---------------------------------------------------------------- Lagrange
def test_lagrange_cumple_condiciones_de_primer_orden():
    asignador, coeficientes = AsignadorEnergia(120), [0.3, 0.5, 0.2, 0.4]
    resultado = asignador.asignar(coeficientes)
    assert sum(resultado.energias) == pytest.approx(120)
    verificacion = asignador.verificar(resultado, coeficientes)
    assert verificacion["max_residuo_dL_dEi"] < 1e-12
    for a, e in zip(coeficientes, resultado.energias):
        assert a / (2 * math.sqrt(e)) == pytest.approx(resultado.multiplicador)


def test_lagrange_es_maximo_frente_a_otros_repartos():
    asignador, coeficientes = AsignadorEnergia(100), [0.7, 0.2, 0.5]
    optimo = asignador.asignar(coeficientes).rendimiento_total
    azar = random.Random(1)
    for _ in range(200):
        pesos = [azar.random() for _ in coeficientes]
        reparto = [100 * p / sum(pesos) for p in pesos]
        assert sum(a * math.sqrt(e) for a, e in zip(coeficientes, reparto)) <= optimo + 1e-9


def test_multiplicador_es_precio_sombra():
    coeficientes = [0.7, 0.2, 0.5]
    base, mas = AsignadorEnergia(100).asignar(coeficientes), AsignadorEnergia(100.01).asignar(coeficientes)
    assert (mas.rendimiento_total - base.rendimiento_total) / 0.01 == pytest.approx(base.multiplicador, rel=1e-3)


def test_lagrange_casos_borde():
    resultado = AsignadorEnergia(50).asignar([0, 0])
    assert resultado.energias == (0.0, 0.0) and resultado.multiplicador == 0
    with pytest.raises(ValueError):
        AsignadorEnergia(0)
    with pytest.raises(ValueError):
        AsignadorEnergia(10).asignar([-1, 2])


def test_derivacion_simbolica():
    condiciones = AsignadorEnergia.derivacion_simbolica(3)["condiciones"]
    assert len(condiciones) == 4  # 3 parciales respecto a E_i y 1 respecto a λ


# ---------------------------------------------------------------- colas
def test_mm1_formulas_conocidas():
    m = calcular_mmc(2, 3, 1)  # ρ = 2/3, Lq = ρ²/(1-ρ), W = 1/(μ-λ)
    assert m.utilizacion == pytest.approx(2 / 3)
    assert m.lq == pytest.approx(4 / 3)
    assert m.w == pytest.approx(1.0)
    assert not calcular_mmc(5, 1, 3).estable


def test_hessiano_orlado_confirma_el_maximo():
    asignador, coeficientes = AsignadorEnergia(120), [0.18, 0.14, 0.21, 0.09]
    hessiano = asignador.hessiano_orlado(asignador.asignar(coeficientes), coeficientes)
    assert hessiano["matriz"].shape == (5, 5)
    assert [m > 0 for m in hessiano["menores"]] == [True, False, True]  # +, −, +
    assert hessiano["es_maximo"]


def test_hessiano_orlado_ignora_clusteres_sin_energia():
    asignador = AsignadorEnergia(50)
    hessiano = asignador.hessiano_orlado(asignador.asignar([0.0, 0.3, 0.4]), [0.0, 0.3, 0.4])
    assert hessiano["matriz"].shape == (3, 3) and hessiano["es_maximo"]


@pytest.mark.parametrize("metodo", ["simbolico", "numerico"])
@pytest.mark.parametrize("semilla", range(5))
def test_los_tres_metodos_de_lagrange_coinciden(metodo, semilla):
    azar = random.Random(semilla)
    coeficientes = [azar.uniform(0.01, 0.3) for _ in range(4)]
    presupuesto = azar.uniform(10, 500)
    analitico = AsignadorEnergia(presupuesto).asignar(coeficientes)
    otro = AsignadorEnergia(presupuesto, metodo).asignar(coeficientes)
    assert otro.metodo == metodo
    assert sum(otro.energias) == pytest.approx(presupuesto)
    assert otro.energias == pytest.approx(analitico.energias, rel=1e-4, abs=1e-6)
    assert otro.multiplicador == pytest.approx(analitico.multiplicador, rel=1e-4)


def test_sp_solve_deduce_la_formula_analitica():
    import sympy as sp

    solucion = AsignadorEnergia.solucion_simbolica(3)
    a1, a2, a3 = sp.symbols("a1:4", positive=True)
    e_max = sp.Symbol("E_max", positive=True)
    esperado = e_max * a1**2 / (a1**2 + a2**2 + a3**2)
    assert sp.simplify(solucion[sp.Symbol("E1", positive=True)] - esperado) == 0
    lam = solucion[sp.Symbol("lambda", positive=True)]
    assert sp.simplify(lam - sp.sqrt(a1**2 + a2**2 + a3**2) / (2 * sp.sqrt(e_max))) == 0


@pytest.mark.parametrize("metodo", ["analitico", "simbolico", "numerico"])
def test_metodos_con_clusteres_sin_rendimiento(metodo):
    resultado = AsignadorEnergia(50, metodo).asignar([0.0, 0.3, 0.4, 0.0])
    assert resultado.energias == pytest.approx((0.0, 18.0, 32.0, 0.0), abs=1e-4)


def test_metodo_de_lagrange_invalido():
    with pytest.raises(ValueError):
        AsignadorEnergia(100, "exacto")
