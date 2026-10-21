"""Pruebas de la configuración y de la simulación SimPy completa."""

import copy
import json

import pytest

from datastream.configuracion import (
    RUTA_CONFIG_DEFECTO,
    ErrorConfiguracion,
    cargar_configuracion,
    construir_configuracion,
)
from datastream.simulacion.centro_datos import SimuladorCentroDatos
from datastream.simulacion.resumen import construir_resumen

DATOS = json.loads(RUTA_CONFIG_DEFECTO.read_text(encoding="utf-8"))


def _modificado(cambio):
    datos = copy.deepcopy(DATOS)
    cambio(datos)
    return datos


# ---------------------------------------------------------------- configuración
def test_configuracion_por_defecto_es_valida():
    config = cargar_configuracion()
    assert len(config.clusters) == 4
    assert config.etapas[0] == ("GW",)


@pytest.mark.parametrize(
    "cambio",
    [
        lambda d: d["simulacion"].update(tasa_llegada=-1),
        lambda d: d["simulacion"].update(duracion="mucho"),
        lambda d: d["simulacion"].update(semilla=1.5),
        lambda d: d["topologia"]["etapas"].__setitem__(0, ["GW", "GW2"]),
        lambda d: d["topologia"]["enlaces"].append({"origen": "GW", "destino": "ZZ", "latencia": 1}),
        lambda d: d["topologia"]["enlaces"].append({"origen": "GW", "destino": "CR1", "latencia": 1}),
        lambda d: d["clusters"][0].update(switch="CR1"),
        lambda d: d["clusters"][1]["servidores"][0].update(id="S1"),
        lambda d: d["clusters"][0]["servidores"][0].update(capacidad=0),
        lambda d: d["tipos_tarea"][0].update(cpu=[5, 2]),
        lambda d: d["tipos_tarea"][3].update(cpu=[6, 99]),
        lambda d: d["temperatura"].update(funcion="x + z"),
        lambda d: d["refrigeracion"].update(nivel_minimo=1.5),
        lambda d: d.pop("energia"),
    ],
)
def test_configuracion_invalida(cambio):
    with pytest.raises(ErrorConfiguracion):
        construir_configuracion(_modificado(cambio))


def test_archivo_inexistente_o_corrupto(tmp_path):
    with pytest.raises(ErrorConfiguracion):
        cargar_configuracion(tmp_path / "no_existe.json")
    roto = tmp_path / "roto.json"
    roto.write_text("{sin cerrar", encoding="utf-8")
    with pytest.raises(ErrorConfiguracion, match="JSON inválido"):
        cargar_configuracion(roto)


# ---------------------------------------------------------------- simulación
@pytest.fixture(scope="module")
def simulador():
    sim = SimuladorCentroDatos(cargar_configuracion().con_cambios(duracion=200))
    sim.ejecutar()
    return sim


def test_conservacion_de_tareas(simulador):
    m = simulador.metricas
    en_sistema = len(simulador.cola) + simulador.tareas_en_transito + simulador.en_proceso
    assert m.generadas == m.procesadas + m.descartadas + en_sistema
    assert m.procesadas > 0


def test_capacidades_nunca_se_exceden(simulador):
    for servidor in simulador.servidores:
        assert 0 <= servidor.en_uso <= servidor.capacidad
        assert servidor.reservado >= 0
        assert servidor.en_uso + servidor.reservado <= servidor.capacidad


def test_presupuesto_y_ventilacion_en_rango(simulador):
    for muestra in simulador.metricas.muestras:
        assert sum(muestra.energias_cluster) == pytest.approx(simulador.config.energia.presupuesto_total)
        assert all(0.1 - 1e-9 <= u <= 1.0 for u in muestra.niveles_ventilador)


def test_simulacion_reproducible():
    config = cargar_configuracion().con_cambios(duracion=60, semilla=123)
    a, b = SimuladorCentroDatos(config), SimuladorCentroDatos(config)
    a.ejecutar()
    b.ejecutar()
    assert (a.metricas.generadas, a.metricas.procesadas) == (b.metricas.generadas, b.metricas.procesadas)


def test_falla_inyectada_provoca_reenrutamiento():
    sim = SimuladorCentroDatos(cargar_configuracion().con_cambios(duracion=60))
    sim.avanzar(5)
    ruta_antes = sim.ruta_de_cluster(0)
    nodo_en_ruta = sim.topologia.nodos[ruta_antes[1]]  # router de núcleo de la ruta actual
    sim._fallar(nodo_en_ruta, manual=True)
    assert nodo_en_ruta.id not in sim.ruta_de_cluster(0)
    assert sim.cambios_ruta[-1].motivo == "falla de nodo"


def test_ajuste_de_tasa_validado():
    sim = SimuladorCentroDatos(cargar_configuracion())
    assert sim.ajustar_tasa_llegada(0.5) == pytest.approx(DATOS["simulacion"]["tasa_llegada"] + 0.5)
    with pytest.raises(ValueError):
        sim.ajustar_tasa_llegada(100)


def test_resumen_serializable(simulador):
    resumen = construir_resumen(simulador)
    texto = json.dumps({k: v for k, v in resumen.items() if k != "series"}, default=str)
    assert "indicadores" in texto and resumen["rutas"]["tabla"]


def test_paquete_se_pierde_si_su_destino_cae_durante_el_viaje():
    sim = SimuladorCentroDatos(cargar_configuracion().con_cambios(duracion=120))
    paquete = None
    while paquete is None and not sim.terminado:
        sim.avanzar(0.25)
        paquete = next((p for p in sim.paquetes if p.destino in {n.id for n in sim.topologia.routers()}), None)
    assert paquete is not None
    nodo, tarea = sim.topologia.nodos[paquete.destino], paquete.tarea
    retransmitidas = sim.metricas.retransmitidas
    sim._fallar(nodo, manual=True)
    sim.avanzar(paquete.inicio + paquete.duracion - sim.ahora + 1e-6)
    assert tarea.reintentos >= 1 and sim.metricas.retransmitidas > retransmitidas
    assert not any(p.tarea is tarea and p.origen == nodo.id for p in sim.paquetes)  # no siguió desde el nodo caído
