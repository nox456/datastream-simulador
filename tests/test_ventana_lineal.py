"""Pruebas de la interfaz gráfica del Simplex (sin abrir una ventana real)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

from datastream import cli, consola  # noqa: E402
from datastream.configuracion import cargar_configuracion  # noqa: E402
from datastream.modelos.programacion_lineal import ProblemaLineal, ResolvedorSimplex, Restriccion  # noqa: E402
from datastream.visualizacion import ventana_lineal  # noqa: E402
from datastream.visualizacion.ventana_lineal import (  # noqa: E402
    ALTO,
    ANCHO,
    VentanaLineal,
    direccion_no_acotada,
    region_factible,
    segmento_en_caja,
    valor_objetivo_de_tabla,
    valores_de_tabla,
)

WYNDOR = ProblemaLineal(
    (3, 5), (Restriccion((1, 0), "<=", 4), Restriccion((0, 2), "<=", 12), Restriccion((3, 2), "<=", 18))
)
TRES_VARIABLES = ProblemaLineal((2, 3, 1), (Restriccion((1, 1, 1), "<=", 10), Restriccion((1, 2, 0), "<=", 12)))


@pytest.fixture
def ventana():
    ventana = VentanaLineal(consola.problemas_lineales(cargar_configuracion(), WYNDOR), "propio")
    yield ventana
    pygame.quit()


def _clic(ventana, punto_lienzo):
    ancho, alto = ventana.ventana.get_size()
    posicion = (round(punto_lienzo[0] * ancho / ANCHO), round(punto_lienzo[1] * alto / ALTO))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=posicion, button=1))
    ventana._procesar_eventos()


def _tecla(ventana, tecla, texto=""):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=tecla, unicode=texto))
    ventana._procesar_eventos()


def _boton(barra, texto):
    return next(b for b in barra.botones if texto in b.etiqueta)


def _misma_figura(poligono, esperados):
    assert len(poligono) == len(esperados)
    for punto in esperados:
        assert any(p == pytest.approx(punto) for p in poligono)


def test_region_factible_del_problema_de_libro():
    _misma_figura(region_factible(WYNDOR, 10, 10), [(0, 0), (4, 0), (4, 3), (2, 6), (0, 6)])


def test_region_factible_con_igualdad_es_un_segmento_y_la_infactible_es_vacia():
    minimizar = consola.problema_ejemplo("minimizar", None)[0]
    segmento = region_factible(minimizar, 5, 5)
    assert all(3 * x + y == pytest.approx(3) for x, y in segmento)
    assert min(x for x, _ in segmento) == pytest.approx(0.4) and max(x for x, _ in segmento) == pytest.approx(0.6)
    assert region_factible(consola.problema_ejemplo("infactible", None)[0], 10, 10) == []


def test_segmento_de_una_recta_dentro_de_la_caja():
    assert segmento_en_caja(3, 2, 18, 10, 10) == (pytest.approx((0, 9)), pytest.approx((6, 0)))
    assert segmento_en_caja(1, 0, 4, 10, 10) == ((4, 0), (4, 10))
    assert segmento_en_caja(1, 1, 50, 10, 10) is None


def test_valores_y_objetivo_de_cada_tabla():
    resultado = ResolvedorSimplex().resolver(WYNDOR)
    recorrido = [valores_de_tabla(t, ("x1", "x2")) for t in resultado.tablas]
    assert recorrido == [(0, 0), (0, 6), (2, 6)]
    assert [valor_objetivo_de_tabla(t) for t in resultado.tablas] == [0, 30, 36]
    minimo = ResolvedorSimplex().resolver(consola.problema_ejemplo("minimizar", None)[0])
    assert valor_objetivo_de_tabla(minimo.tablas[-1]) == pytest.approx(3.4)


def test_direccion_de_la_solucion_no_acotada():
    resultado = ResolvedorSimplex().resolver(consola.problema_ejemplo("no-acotado", None)[0])
    assert direccion_no_acotada(resultado.tablas[-1], ("x1", "x2")) == pytest.approx((0, 1))
    assert direccion_no_acotada(resultado.tablas[0], ("x1", "x2")) is None


def test_los_botones_caben_sin_superponerse(ventana):
    for barra in (ventana.barra_ejemplos, ventana.barra_pasos):
        rects = [b.rect for b in barra.botones]
        assert all(barra.area.contains(r) for r in rects)
        assert all(a.right < b.left for a, b in zip(rects, rects[1:]))
    assert len(ventana.barra_ejemplos.botones) == 6


def test_navegacion_con_botones_y_teclas(ventana):
    assert ventana.paso == 0
    _clic(ventana, _boton(ventana.barra_pasos, "Siguiente").rect.center)
    _tecla(ventana, pygame.K_RIGHT)
    assert ventana.paso == 2
    _tecla(ventana, pygame.K_RIGHT)
    assert ventana.paso == 2 and "última" in ventana._mensaje[0]
    _tecla(ventana, pygame.K_HOME)
    assert ventana.paso == 0
    _clic(ventana, _boton(ventana.barra_pasos, "Anterior").rect.center)
    assert ventana.paso == 0 and "primera" in ventana._mensaje[0]
    _clic(ventana, _boton(ventana.barra_pasos, "Final").rect.center)
    assert ventana.paso == len(ventana.pasos) - 1
    _tecla(ventana, pygame.K_x, "x")
    assert "no reconocido" in ventana._mensaje[0]
    _tecla(ventana, pygame.K_ESCAPE)
    assert not ventana.activo


def test_cambiar_de_ejemplo_con_boton_y_tecla(ventana):
    _clic(ventana, _boton(ventana.barra_ejemplos, "Parcial IV").rect.center)
    assert ventana.nombre == "parcial-iv"
    assert len(ventana.pasos) == len(ventana.resultado.tablas) + 1
    ventana.ir_a(len(ventana.pasos) - 1)
    assert ventana.es_alternativa
    _tecla(ventana, pygame.K_5)
    assert ventana.nombre == "infactible" and ventana.paso == 0


@pytest.mark.parametrize("nombre", [*consola.EJEMPLOS_LINEAL, "propio"])
def test_dibuja_todas_las_tablas_de_cada_ejemplo(ventana, nombre):
    ventana.seleccionar(nombre)
    for paso in range(len(ventana.pasos)):
        ventana.ir_a(paso)
        ventana.dibujar()


def test_dibuja_barras_con_mas_de_dos_variables():
    ventana = VentanaLineal(consola.problemas_lineales(cargar_configuracion(), TRES_VARIABLES), "propio")
    try:
        llamadas = []
        original = ventana._dibujar_barras
        ventana._dibujar_barras = lambda: (llamadas.append(1), original())
        for paso in range(len(ventana.pasos)):
            ventana.ir_a(paso)
            ventana.dibujar()
        assert len(llamadas) == len(ventana.pasos)
    finally:
        pygame.quit()


def test_problema_inicial_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        VentanaLineal(consola.problemas_lineales(cargar_configuracion()), "propio")


def test_comando_lineal_abre_la_interfaz(monkeypatch):
    abiertas = []
    monkeypatch.setattr(ventana_lineal.VentanaLineal, "ejecutar", lambda self: abiertas.append(self))
    assert cli.main(["lineal", "--gui", "--ejemplo", "minimizar"]) == 0
    assert cli.main(["lineal", "--gui", "--objetivo", "3,5", "--restriccion", "1,0<=4", "--restriccion", "3,2<=18"]) == 0
    assert [v.nombre for v in abiertas] == ["minimizar", "propio"]
    assert abiertas[1].problema.objetivo == (3, 5)
    pygame.quit()


def test_menu_ofrece_la_interfaz(monkeypatch):
    abiertas = []
    monkeypatch.setattr(ventana_lineal.VentanaLineal, "ejecutar", lambda self: abiertas.append(self.nombre))
    respuestas = iter(["10", "no-acotado", "s", "", "0"])
    monkeypatch.setattr("builtins.input", lambda _mensaje="": next(respuestas))
    assert cli.menu_interactivo(cargar_configuracion(), None) == 0
    assert abiertas == ["no-acotado"]
    pygame.quit()
