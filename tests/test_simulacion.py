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
from datastream.modelos.presupuesto import AsignadorEnergia
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


def test_cluster_inalcanzable_se_informa_claramente():
    sim = SimuladorCentroDatos(cargar_configuracion().con_cambios(duracion=60))
    sim.avanzar(2)
    switch = sim.topologia.nodos[sim.clusters[0].switch]
    sim._fallar(switch, manual=True)
    assert sim.ruta_de_cluster(0) is None
    assert sim.cambios_ruta[-1].motivo == "clúster inalcanzable"
    ultimo = [e for e in sim.metricas.eventos if e.tipo == "ruta"][-1].mensaje
    assert "inalcanzable" in ultimo and "sin ruta (" not in ultimo


def test_texto_ia_conserva_negritas_al_inicio_de_linea():
    from datastream.reportes.generador_pdf import GeneradorReportePDF

    parrafos = GeneradorReportePDF()._texto_ia("## Título\n**Nota:** texto\n- viñeta **x**\n* otra viñeta")
    textos = [p.text for p in parrafos]
    assert textos == ["Título", "<b>Nota:</b> texto", "viñeta <b>x</b>", "otra viñeta"]
    assert all("*" not in t for t in textos)


def test_error_generico_de_la_api_usa_el_analisis_local(simulador, monkeypatch):
    import openai

    from datastream.reportes import analista_ia

    def falla(self, resumen):
        raise openai.APIError("respuesta inesperada", request=None, body=None)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-prueba")
    monkeypatch.setattr(analista_ia.AnalistaDeepSeek, "analizar", falla)
    resultado = analista_ia.analizar_resultados(construir_resumen(simulador))
    assert resultado.fuente == "reglas locales" and "error de la API" in resultado.texto


def test_respuesta_cortada_de_la_ia_usa_el_analisis_local(simulador, monkeypatch):
    from types import SimpleNamespace

    from datastream.reportes import analista_ia

    cortada = SimpleNamespace(
        model="deepseek-chat",
        choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="## Diagnóstico gen"))],
    )
    cliente = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: cortada)))

    def iniciar(self, modelo=None):
        self.modelo, self.cliente = "deepseek-chat", cliente

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-prueba")
    monkeypatch.setattr(analista_ia.AnalistaDeepSeek, "__init__", iniciar)
    resultado = analista_ia.analizar_resultados(construir_resumen(simulador))
    assert resultado.fuente == "reglas locales" and "incompleta" in resultado.texto


def test_resumen_incluye_el_hessiano_orlado(simulador):
    verificacion = construir_resumen(simulador)["energia"]["verificacion"]
    menores = verificacion["menores_hessiano_orlado"]
    assert len(menores) == len(simulador.clusters) - 1  # órdenes 3 a n+1
    assert [m > 0 for m in menores] == [i % 2 == 0 for i in range(len(menores))]  # +, −, +, …
    assert verificacion["es_maximo"]


def test_resumen_distingue_ruta_optima_y_ruta_en_uso(simulador):
    resumen = construir_resumen(simulador)
    for fila in resumen["rutas"]["por_cluster"]:
        if fila["latencia_optima"] is not None and fila["latencia"] is not None:
            # la ruta en uso nunca es más de un 8 % peor que la óptima de la DP
            assert fila["latencia"] <= fila["latencia_optima"] * (1 + resumen["rutas"]["umbral_cambio_ruta"]) + 1e-9


def test_resumen_muestra_una_mochila_representativa(simulador):
    destacada = construir_resumen(simulador)["mochila"]["destacada"]
    assert destacada and 1 < destacada["total_articulos"] <= 15
    elegidas = [a for a in destacada["articulos"] if a["elegida"]]
    caben = [a for a in destacada["articulos"] if a["peso"] <= destacada["capacidad"]]
    assert 2 <= len(elegidas) < len(caben)  # el modelo realmente tuvo que decidir


def test_la_ia_recibe_el_estado_final_separado_de_los_maximos(simulador):
    from datastream.reportes import analista_ia

    resumen = construir_resumen(simulador)
    final = resumen["clusters_al_final"]
    assert len(final) == len(simulador.clusters) and "temperatura_al_final" in final[0]
    assert resumen["termico"]["temperatura_maxima_observada"] >= max(c["temperatura_al_final"] for c in final)
    # Las claves que las instrucciones mencionan deben existir en el JSON que recibe la IA.
    enviado = analista_ia._resumen_para_ia(resumen)
    for clave in ("clusters_al_final", "temperatura_maxima_observada", "temperatura_emergencia", "destacada"):
        assert clave in analista_ia.INSTRUCCIONES_SISTEMA and f'"{clave}"' in enviado


RUTA_CONFIG_BELLMAN_FORD = RUTA_CONFIG_DEFECTO.with_name("configuracion_bellman_ford.json")


def test_por_defecto_se_usa_la_dp_por_etapas():
    assert cargar_configuracion().algoritmo_rutas == "etapas"
    assert cargar_configuracion(RUTA_CONFIG_BELLMAN_FORD).algoritmo_rutas == "bellman_ford"


def test_enlaces_dentro_de_una_etapa_solo_con_bellman_ford():
    lateral = {"origen": "CR1", "destino": "CR2", "latencia": 0.3}
    with pytest.raises(ErrorConfiguracion, match="bellman_ford"):
        construir_configuracion(_modificado(lambda d: d["topologia"]["enlaces"].append(lateral)))

    def con_bellman_ford(d):
        d["topologia"]["enlaces"].append(lateral)
        d["topologia"]["algoritmo"] = "bellman_ford"

    assert construir_configuracion(_modificado(con_bellman_ford)).algoritmo_rutas == "bellman_ford"


@pytest.mark.parametrize(
    "cambio",
    [
        lambda d: d["topologia"].update(algoritmo="dijkstra"),
        lambda d: (d["topologia"].update(algoritmo="bellman_ford"),
                   d["topologia"]["enlaces"].append({"origen": "CR1", "destino": "CR1", "latencia": 1})),
    ],
)
def test_algoritmo_o_enlaces_invalidos(cambio):
    with pytest.raises(ErrorConfiguracion):
        construir_configuracion(_modificado(cambio))


def test_con_algoritmo_comprueba_la_topologia():
    config = cargar_configuracion(RUTA_CONFIG_BELLMAN_FORD)
    with pytest.raises(ErrorConfiguracion, match="no une etapas consecutivas"):
        config.con_algoritmo("etapas")
    assert cargar_configuracion().con_algoritmo("bellman_ford").algoritmo_rutas == "bellman_ford"


def test_simulacion_con_bellman_ford_usa_los_enlaces_laterales():
    sim = SimuladorCentroDatos(cargar_configuracion(RUTA_CONFIG_BELLMAN_FORD).con_cambios(duracion=200))
    sim.ejecutar()
    m = sim.metricas
    assert m.generadas == m.procesadas + m.descartadas + len(sim.cola) + sim.tareas_en_transito + sim.en_proceso
    laterales = {(e.origen, e.destino) for e in sim.config.enlaces if e.origen[:2] == e.destino[:2]}
    usados = {t for c in sim.cambios_ruta if c.nueva for t in zip(c.nueva, c.nueva[1:])}
    assert usados & laterales
    assert construir_resumen(sim)["rutas"]["algoritmo"] == "bellman_ford"


def test_ambos_algoritmos_dan_las_mismas_rutas_en_la_red_por_etapas():
    por_etapas = SimuladorCentroDatos(cargar_configuracion())
    bellman = SimuladorCentroDatos(cargar_configuracion().con_algoritmo("bellman_ford"))
    assert por_etapas.rutas_optimas == bellman.rutas_optimas


def test_metodo_de_lagrange_en_la_configuracion():
    assert cargar_configuracion().energia.metodo == "analitico"
    assert construir_configuracion(_modificado(lambda d: d["energia"].update(metodo="numerico"))).energia.metodo == "numerico"
    with pytest.raises(ErrorConfiguracion, match="energia.metodo"):
        construir_configuracion(_modificado(lambda d: d["energia"].update(metodo="exacto")))
    with pytest.raises(ErrorConfiguracion):
        cargar_configuracion().con_metodo_lagrange("exacto")


def test_simulacion_con_lagrange_simbolico_es_identica_a_la_analitica():
    config = cargar_configuracion().con_cambios(duracion=60)
    analitica, simbolica = SimuladorCentroDatos(config), SimuladorCentroDatos(config.con_metodo_lagrange("simbolico"))
    analitica.ejecutar()
    simbolica.ejecutar()
    assert (analitica.metricas.procesadas, analitica.metricas.descartadas) == (simbolica.metricas.procesadas, simbolica.metricas.descartadas)
    assert construir_resumen(simbolica)["energia"]["metodo"] == "simbolico"


def test_simulacion_con_lagrange_numerico_respeta_el_presupuesto():
    sim = SimuladorCentroDatos(cargar_configuracion().con_cambios(duracion=60).con_metodo_lagrange("numerico"))
    sim.ejecutar()
    for muestra in sim.metricas.muestras:
        assert sum(muestra.energias_cluster) == pytest.approx(sim.config.energia.presupuesto_total)
    esperado = AsignadorEnergia(sim.config.energia.presupuesto_total).asignar(sim.ultimos_coeficientes)
    assert sim.ultimo_lagrange.energias == pytest.approx(esperado.energias, rel=1e-4, abs=1e-6)


def test_ajuste_del_presupuesto_en_vivo():
    sim = SimuladorCentroDatos(cargar_configuracion().con_cambios(duracion=60))
    sim.avanzar(20)
    lambda_antes = sim.ultimo_lagrange.multiplicador
    velocidad_antes = sum(c.factor_velocidad for c in sim.clusters)
    assert sim.ajustar_presupuesto(60) == pytest.approx(180)
    assert sum(sim.ultimo_lagrange.energias) == pytest.approx(180)
    assert sim.ultimo_lagrange.multiplicador < lambda_antes
    assert sum(c.factor_velocidad for c in sim.clusters) > velocidad_antes
    assert construir_resumen(sim)["parametros"]["presupuesto_electrico"] == pytest.approx(180)
    sim.avanzar(5)
    assert sum(sim.metricas.muestras[-1].energias_cluster) == pytest.approx(180)
    with pytest.raises(ValueError):
        sim.ajustar_presupuesto(1000)
    with pytest.raises(ValueError):
        sim.ajustar_presupuesto(-170)


def test_ajuste_del_presupuesto_conserva_el_metodo_de_lagrange():
    sim = SimuladorCentroDatos(cargar_configuracion().con_metodo_lagrange("numerico"))
    sim.ajustar_presupuesto(12)
    assert sim.ultimo_lagrange.metodo == "numerico"


def test_ajuste_del_umbral_de_emergencia_en_vivo():
    sim = SimuladorCentroDatos(cargar_configuracion().con_cambios(duracion=60))
    critico = sim.analisis_termico.carga_emergencia
    assert sim.ajustar_temperatura_emergencia(-15) == pytest.approx(60)
    assert sim.analisis_termico.carga_emergencia < critico
    assert sim.modelo_termico.temperatura(sim.analisis_termico.carga_emergencia) == pytest.approx(60, abs=1e-6)
    assert construir_resumen(sim)["termico"]["temperatura_emergencia"] == pytest.approx(60)
    sim.ajustar_temperatura_emergencia(+60)
    assert sim.analisis_termico.carga_emergencia is None
    with pytest.raises(ValueError):
        sim.ajustar_temperatura_emergencia(+100)
    with pytest.raises(ValueError):
        sim.ajustar_temperatura_emergencia(-100)


def test_umbral_bajo_activa_emergencias_en_la_simulacion():
    sim = SimuladorCentroDatos(cargar_configuracion().con_cambios(duracion=300))
    sim.ajustar_temperatura_emergencia(-35)
    sim.ejecutar()
    assert any(e.tipo == "emergencia" for e in sim.metricas.eventos)
