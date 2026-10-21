"""Pruebas del menú interactivo y de la línea de comandos."""

import pytest

from datastream import cli, consola
from datastream.configuracion import cargar_configuracion


def _responder(monkeypatch, respuestas):
    """Simula lo que el usuario escribe en cada input() del menú."""
    entradas = iter(respuestas)
    monkeypatch.setattr("builtins.input", lambda _mensaje="": next(entradas))


def test_menu_sigue_abierto_tras_un_error_inesperado(monkeypatch, capsys):
    def falla(*_args, **_kwargs):
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(consola, "demo_temperatura", falla)
    # opción 5 → Enter (umbral por defecto) → Enter (volver al menú) → 0 (salir)
    _responder(monkeypatch, ["5", "", "", "0"])
    assert cli.menu_interactivo(cargar_configuracion(), None) == 0
    assert "error inesperado: RuntimeError: fallo simulado" in capsys.readouterr().out


def test_ctrl_c_en_el_menu_no_se_trata_como_error(monkeypatch):
    def interrumpe(_mensaje=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrumpe)
    with pytest.raises(KeyboardInterrupt):
        cli.menu_interactivo(cargar_configuracion(), None)


@pytest.mark.parametrize(
    "argumentos",
    [
        ["ruta", "--destino", "GW"],  # el gateway es el origen
        ["ruta", "--destino", "XX"],  # nodo inexistente
        ["ruta", "--falla", "TR4", "--destino", "TR4"],  # destino caído
    ],
)
def test_destino_invalido_de_la_recursion_hacia_atras(argumentos, capsys):
    assert cli.main(argumentos) == 2
    assert "Error de datos" in capsys.readouterr().err


def test_ruta_muestra_ambas_recursiones(capsys):
    assert cli.main(["ruta", "--destino", "TR4"]) == 0
    salida = capsys.readouterr().out
    assert "Recursión hacia adelante" in salida and "Recursión hacia atrás" in salida
    assert "Ambas recursiones dan la misma latencia óptima." in salida


def test_ejemplos_de_clase_por_linea_de_comandos(capsys):
    assert cli.main(["ruta", "--ejemplo-clase"]) == 0
    assert "Ruta más corta: 1 > 4 > 5 > 7 con 21 km" in capsys.readouterr().out
    assert cli.main(["mochila", "--ejemplo-clase"]) == 0
    assert "2 de Artículo 1, 0 de Artículo 2, 0 de Artículo 3 · rendimiento 62" in capsys.readouterr().out


def test_opcion_9_del_menu_muestra_los_ejemplos_de_clase(monkeypatch, capsys):
    _responder(monkeypatch, ["9", "", "0"])  # opción 9 → Enter (volver al menú) → 0 (salir)
    assert cli.menu_interactivo(cargar_configuracion(), None) == 0
    salida = capsys.readouterr().out
    assert "1 > 4 > 5 > 7 con 21 km" in salida and "rendimiento 62" in salida


def test_ruta_con_bellman_ford_por_linea_de_comandos(capsys):
    assert cli.main(["ruta", "--algoritmo", "bellman-ford"]) == 0
    salida = capsys.readouterr().out
    assert "Bellman-Ford" in salida and "La DP por etapas da las mismas latencias óptimas." in salida


@pytest.mark.parametrize(
    "argumentos",
    [
        ["--config", "datos/configuracion_bellman_ford.json", "ruta", "--algoritmo", "etapas"],
        ["ruta", "--algoritmo", "bellman-ford", "--destino", "TR4"],
    ],
)
def test_combinaciones_invalidas_de_algoritmo(argumentos, capsys):
    assert cli.main(argumentos) == 2
    assert "Error de datos" in capsys.readouterr().err


def test_menu_pregunta_el_algoritmo_y_lo_valida(monkeypatch, capsys):
    _responder(monkeypatch, ["3", "dijkstra", "bellman-ford", "", "", "0"])
    assert cli.menu_interactivo(cargar_configuracion(), None) == 0
    salida = capsys.readouterr().out
    assert "Escriba 'etapas' o 'bellman-ford'" in salida and "Bellman-Ford" in salida


def test_lagrange_por_metodo_simbolico_y_numerico(capsys):
    assert cli.main(["lagrange", "--metodo", "simbolico"]) == 0
    salida = capsys.readouterr().out
    assert "sp.solve resolvió el sistema" in salida and "Comparado con la fórmula analítica" in salida
    assert cli.main(["lagrange", "--metodo", "numerico"]) == 0
    assert "SciPy (SLSQP)" in capsys.readouterr().out


def test_menu_pregunta_el_metodo_de_lagrange(monkeypatch, capsys):
    _responder(monkeypatch, ["7", "", "", "exacto", "numerico", "", "0"])
    assert cli.menu_interactivo(cargar_configuracion(), None) == 0
    salida = capsys.readouterr().out
    assert "Escriba 'analitico', 'simbolico' o 'numerico'" in salida and "SciPy (SLSQP)" in salida
