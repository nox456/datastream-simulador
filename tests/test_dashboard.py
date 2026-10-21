"""Pruebas de los controles del dashboard (sin abrir una ventana real)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # Pygame sin pantalla

import pygame  # noqa: E402
import pytest  # noqa: E402

from datastream.configuracion import cargar_configuracion  # noqa: E402
from datastream.simulacion.centro_datos import SimuladorCentroDatos  # noqa: E402
from datastream.visualizacion.dashboard import ALTO, ANCHO, Dashboard  # noqa: E402


@pytest.fixture
def dashboard():
    tablero = Dashboard(SimuladorCentroDatos(cargar_configuracion().con_cambios(duracion=60)))
    yield tablero
    pygame.quit()


def _clic(tablero, punto_lienzo, boton=1):
    """Hace clic en un punto del lienzo, convirtiéndolo a coordenadas de la ventana (que puede estar escalada)."""
    ancho, alto = tablero.ventana.get_size()
    posicion = (round(punto_lienzo[0] * ancho / ANCHO), round(punto_lienzo[1] * alto / ALTO))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=posicion, button=boton))
    tablero._procesar_eventos()


def _boton(tablero, texto):
    return next(b for b in tablero.barra.botones if texto in b.etiqueta)


def test_los_botones_caben_en_el_encabezado_sin_superponerse(dashboard):
    rects = [b.rect for b in dashboard.barra.botones]
    assert len(rects) == 8
    assert all(dashboard.barra.area.contains(r) for r in rects)
    assert all(a.right < b.left for a, b in zip(rects, rects[1:]))


def test_boton_pausa_alterna_y_cambia_su_texto(dashboard):
    _clic(dashboard, _boton(dashboard, "Pausa").rect.center)
    assert dashboard.pausado and "Reanudar" in dashboard.barra.botones[0].etiqueta
    _clic(dashboard, dashboard.barra.botones[0].rect.center)
    assert not dashboard.pausado


def test_botones_de_velocidad_tasa_y_salir(dashboard):
    velocidad, tasa = dashboard.velocidad, dashboard.sim.tasa_llegada
    _clic(dashboard, _boton(dashboard, "Velocidad +").rect.center)
    _clic(dashboard, _boton(dashboard, "Tasa +").rect.center)
    assert dashboard.velocidad > velocidad and dashboard.sim.tasa_llegada == pytest.approx(tasa + 0.5)
    _clic(dashboard, _boton(dashboard, "Salir").rect.center)
    assert not dashboard.activo


def test_boton_falla_inyecta_una_falla(dashboard):
    dashboard.sim.avanzar(2)
    fallas = dashboard.sim.metricas.fallas
    _clic(dashboard, _boton(dashboard, "Falla").rect.center)
    assert dashboard.sim.metricas.fallas == fallas + 1


def test_clic_en_un_cluster_selecciona_su_ruta(dashboard):
    _clic(dashboard, dashboard.topologia._marco_cluster(2).center)
    assert dashboard.topologia.cluster_seleccionado == 2


def test_clic_derecho_o_fuera_de_los_botones_no_hace_nada(dashboard):
    _clic(dashboard, _boton(dashboard, "Salir").rect.center, boton=3)
    _clic(dashboard, (700, 300))  # zona de la topología sin botones ni clústeres
    assert dashboard.activo and not dashboard.pausado and dashboard.topologia.cluster_seleccionado == 0


def test_los_clics_funcionan_con_la_ventana_escalada(dashboard):
    dashboard.ventana = pygame.display.set_mode((ANCHO // 2, ALTO // 2))
    _clic(dashboard, _boton(dashboard, "Pausa").rect.center)
    assert dashboard.pausado
