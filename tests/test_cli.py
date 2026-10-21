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
