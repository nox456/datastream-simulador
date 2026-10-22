"""Pruebas de programación lineal: método Simplex de dos fases, casos especiales y precios sombra."""

import random

import pytest

from datastream import cli, consola
from datastream.configuracion import cargar_configuracion
from datastream.modelos.programacion_lineal import (
    ProblemaLineal,
    ResolvedorSimplex,
    Restriccion,
    resolver_con_scipy,
    validar_problema,
)

WYNDOR = ProblemaLineal(
    (3, 5), (Restriccion((1, 0), "<=", 4), Restriccion((0, 2), "<=", 12), Restriccion((3, 2), "<=", 18))
)


def _resolver(problema):
    return ResolvedorSimplex().resolver(problema)


def _con_lado_derecho(problema, i, delta):
    restricciones = list(problema.restricciones)
    r = restricciones[i]
    restricciones[i] = Restriccion(r.coeficientes, r.sentido, r.lado_derecho + delta)
    return ProblemaLineal(problema.objetivo, tuple(restricciones), problema.sentido, problema.variables)


def test_problema_de_libro_con_cada_tabla():
    resultado = _resolver(WYNDOR)
    assert resultado.es_optimo and not resultado.usa_dos_fases
    assert resultado.valores == pytest.approx((2, 6))
    assert resultado.valor_objetivo == pytest.approx(36)
    assert resultado.precios_sombra == pytest.approx((0, 1.5, 1))
    assert resultado.holguras == pytest.approx((2, 0, 0))
    primera = resultado.tablas[0]
    assert primera.base == ("s1", "s2", "s3")
    assert primera.fila_objetivo == pytest.approx((-3, -5, 0, 0, 0, 0))
    assert (primera.entrante, primera.saliente) == ("x2", "s2")
    assert primera.razones == (None, 6, 9)
    assert [t.entrante for t in resultado.tablas] == ["x2", "x1", None]


def test_cada_tabla_tiene_la_identidad_en_las_columnas_basicas():
    for problema in (WYNDOR, consola.problema_ejemplo("minimizar", None)[0]):
        for tabla in _resolver(problema).tablas:
            for i, variable in enumerate(tabla.base):
                k = tabla.columnas.index(variable)
                assert [fila[k] for fila in tabla.filas] == pytest.approx([1.0 if j == i else 0.0 for j in range(len(tabla.base))])
                assert tabla.fila_objetivo[k] == pytest.approx(0)


def test_ejemplo_del_centro_de_datos():
    problema = consola.problema_centro_datos(cargar_configuracion())
    assert problema.restricciones[0].lado_derecho == 180
    assert problema.restricciones[1].lado_derecho == 120
    resultado = _resolver(problema)
    assert resultado.usa_dos_fases
    assert dict(zip(resultado.variables, resultado.valores)) == pytest.approx(
        {"web": 10, "base_datos": 7.5, "video": 0, "ia": 16.25}
    )
    assert resultado.valor_objetivo == pytest.approx(1290)
    assert resultado.precios_sombra == pytest.approx((4, 5, -3))


def test_minimizacion_con_igualdad_y_dos_fases():
    resultado = _resolver(consola.problema_ejemplo("minimizar", None)[0])
    assert resultado.usa_dos_fases
    assert resultado.valores == pytest.approx((0.4, 1.8))
    assert resultado.valor_objetivo == pytest.approx(3.4)
    assert resultado.precios_sombra == pytest.approx((1.4, 0, -0.2))
    assert resultado.tablas[-1].etiqueta_objetivo == "-Z"


@pytest.mark.parametrize("nombre", ["centro-datos", "minimizar"])
def test_precio_sombra_igual_a_la_derivada_numerica(nombre):
    problema = consola.problema_ejemplo(nombre, cargar_configuracion())[0]
    base = _resolver(problema)
    delta = 1e-3
    for i, precio in enumerate(base.precios_sombra):
        _, _, z = resolver_con_scipy(_con_lado_derecho(problema, i, delta))
        assert (z - base.valor_objetivo) / delta == pytest.approx(precio, abs=1e-5)


def test_soluciones_multiples_del_parcial():
    resultado = _resolver(consola.problema_ejemplo("parcial-iv", None)[0])
    assert resultado.valor_objetivo == pytest.approx(150)
    assert resultado.valores == pytest.approx((1.5, 0))
    assert resultado.soluciones_multiples
    assert resultado.pivote_alternativo == ("x2", "x1")
    assert resultado.solucion_alternativa == pytest.approx((0, 3.75))
    assert 100 * 0 + 40 * 3.75 == pytest.approx(resultado.valor_objetivo)


def test_solucion_unica_no_se_declara_multiple():
    assert not _resolver(WYNDOR).soluciones_multiples


def test_soluciones_multiples_a_lo_largo_de_una_semirrecta():
    problema = ProblemaLineal((-1, 1), (Restriccion((1, -1), "<=", 1),), "min")
    resultado = _resolver(problema)
    assert resultado.valor_objetivo == pytest.approx(-1)
    assert resultado.soluciones_multiples and resultado.solucion_alternativa is None


def test_solucion_no_acotada():
    resultado = _resolver(consola.problema_ejemplo("no-acotado", None)[0])
    assert resultado.estado == "no_acotado"
    assert resultado.variable_no_acotada == "s1"
    assert resultado.tablas[-1].saliente is None
    assert resultado.valores == () and resultado.valor_objetivo is None


def test_problema_infactible():
    resultado = _resolver(consola.problema_ejemplo("infactible", None)[0])
    assert resultado.estado == "infactible"
    assert all(t.fase == 1 for t in resultado.tablas)
    assert resultado.tablas[-1].fila_objetivo[-1] == pytest.approx(-4)


def test_lado_derecho_negativo_se_normaliza():
    problema = ProblemaLineal((1, 1), (Restriccion((-1, -1), ">=", -4),))
    resultado = _resolver(problema)
    assert resultado.valor_objetivo == pytest.approx(4)
    assert resultado.precios_sombra == pytest.approx((-1,))
    assert resultado.holguras == pytest.approx((0,))


def test_restriccion_redundante():
    problema = ProblemaLineal((1, 0), (Restriccion((1, 1), "=", 2), Restriccion((2, 2), "=", 4)))
    resultado = _resolver(problema)
    assert resultado.valores == pytest.approx((2, 0))
    assert resultado.restricciones_redundantes == (1,)
    assert resultado.precios_sombra == pytest.approx((1, 0))


def test_ejemplo_de_beale_no_cicla():
    problema = ProblemaLineal(
        (0.75, -20, 0.5, -6),
        (
            Restriccion((0.25, -8, -1, 9), "<=", 0),
            Restriccion((0.5, -12, -0.5, 3), "<=", 0),
            Restriccion((0, 0, 1, 0), "<=", 1),
        ),
    )
    resultado = _resolver(problema)
    assert resultado.tablas[6].base == resultado.tablas[0].base
    assert resultado.es_optimo
    assert resultado.valor_objetivo == pytest.approx(1.25)
    assert resultado.valores == pytest.approx((1, 0, 1, 0))


def test_coincide_con_scipy_en_problemas_aleatorios():
    azar = random.Random(2026)
    estados = set()
    for _ in range(300):
        n, m = azar.randint(2, 4), azar.randint(1, 4)
        problema = ProblemaLineal(
            tuple(azar.randint(-5, 9) for _ in range(n)),
            tuple(
                Restriccion(tuple(azar.randint(-3, 6) for _ in range(n)), azar.choice(["<=", "<=", ">=", "="]),
                            azar.randint(-5, 20))
                for _ in range(m)
            ),
            azar.choice(["max", "min"]),
        )
        resultado = _resolver(problema)
        estado, _, z = resolver_con_scipy(problema)
        estados.add(estado)
        assert resultado.estado == estado
        if estado == "optimo":
            assert resultado.valor_objetivo == pytest.approx(z, abs=1e-7)
            for r, holgura in zip(problema.restricciones, resultado.holguras):
                assert holgura >= -1e-9 if r.sentido != "=" else holgura == pytest.approx(0, abs=1e-9)
    assert estados == {"optimo", "no_acotado", "infactible"}


@pytest.mark.parametrize(
    "problema, mensaje",
    [
        (ProblemaLineal((1,), (Restriccion((1,), "<=", 1),), "maximo"), "'max' o 'min'"),
        (ProblemaLineal((), (Restriccion((), "<=", 1),)), "al menos una variable"),
        (ProblemaLineal((1, 2), ()), "al menos una restricción"),
        (ProblemaLineal((1, 2), (Restriccion((1,), "<=", 1),)), "1 coeficientes"),
        (ProblemaLineal((1,), (Restriccion((1,), "<", 1),)), "sentido inválido"),
        (ProblemaLineal((float("nan"),), (Restriccion((1,), "<=", 1),)), "números finitos"),
        (ProblemaLineal((1,), (Restriccion((1,), "<=", 1),), "max", ("a", "b")), "nombres"),
    ],
)
def test_validacion_del_problema(problema, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        validar_problema(problema)


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("1,2<=4", Restriccion((1, 2), "<=", 4)),
        (" 3, -1 >= 2.5 ", Restriccion((3, -1), ">=", 2.5)),
        ("1,1=5", Restriccion((1, 1), "=", 5)),
        ("2,1≤8", Restriccion((2, 1), "<=", 8)),
        ("2,1=>-3", Restriccion((2, 1), ">=", -3)),
    ],
)
def test_lectura_de_restricciones(texto, esperado):
    assert cli.leer_restriccion(texto) == esperado


@pytest.mark.parametrize("texto", ["1,2<4", "1,a<=3", "1,2<=3,4", "<=3", "1,2<=inf"])
def test_restricciones_mal_escritas(texto):
    with pytest.raises(ValueError):
        cli.leer_restriccion(texto)


def test_comando_lineal_por_defecto_y_con_problema_propio(capsys):
    assert cli.main(["lineal"]) == 0
    salida = capsys.readouterr().out
    assert "Fase 1 · Tabla 0" in salida and "Z* = 1290" in salida and "coincide con el Simplex" in salida
    argumentos = ["lineal", "--objetivo", "3,5", "--restriccion", "1,0<=4", "--restriccion", "0,2<=12",
                  "--restriccion", "3,2<=18"]
    assert cli.main(argumentos) == 0
    assert "Z* = 36" in capsys.readouterr().out


@pytest.mark.parametrize("nombre, texto", [
    ("parcial-iv", "SOLUCIONES MÚLTIPLES"),
    ("no-acotado", "SOLUCIÓN NO ACOTADA"),
    ("infactible", "PROBLEMA INFACTIBLE"),
    ("minimizar", "Z* = 3.4"),
])
def test_ejemplos_informan_cada_caso(nombre, texto, capsys):
    assert cli.main(["lineal", "--ejemplo", nombre]) == 0
    salida = capsys.readouterr().out
    assert texto in salida and "ATENCIÓN" not in salida


@pytest.mark.parametrize(
    "argumentos",
    [
        ["lineal", "--objetivo", "3,5"],
        ["lineal", "--restriccion", "1,0<=4"],
        ["lineal", "--minimizar"],
        ["lineal", "--objetivo", "3,5", "--restriccion", "1<=4"],
        ["lineal", "--ejemplo", "infactible", "--objetivo", "1,1", "--restriccion", "1,1<=2"],
    ],
)
def test_comando_lineal_con_datos_invalidos(argumentos, capsys):
    assert cli.main(argumentos) == 2
    assert "Error de datos" in capsys.readouterr().err


def test_menu_resuelve_un_problema_propio_y_repite_los_datos_invalidos(monkeypatch, capsys):
    respuestas = iter(["10", "propio", "min", "2,x", "2,3", "2", "1,1>4", "1,1,1>=4", "1,1>=4", "1,3>=6", "n", "", "0"])
    monkeypatch.setattr("builtins.input", lambda _mensaje="": next(respuestas))
    assert cli.menu_interactivo(cargar_configuracion(), None) == 0
    salida = capsys.readouterr().out
    assert salida.count("✗") == 3
    assert "Z* = 9" in salida
