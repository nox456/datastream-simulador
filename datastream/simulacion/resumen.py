"""Resumen serializable del estado de la simulación (lo consumen la IA y el PDF)."""

from __future__ import annotations

from datetime import datetime

from ..modelos.colas import calcular_mmc, ley_de_little
from .centro_datos import UMBRAL_CAMBIO_RUTA, SimuladorCentroDatos


def _ruta(ruta) -> str:
    return " > ".join(ruta) if ruta else "sin ruta"


def construir_resumen(sim: SimuladorCentroDatos) -> dict:
    """Copia de datos planos: puede usarse desde otro hilo sin tocar la simulación."""
    m = sim.metricas
    cfg = sim.config
    t = max(sim.ahora, 1e-9)
    nombres = [c.nombre for c in sim.clusters]

    # --- Teoría de colas: referencia M/M/c y ley de Little ------------------
    tipos = cfg.tipos_tarea
    servicio_medio = sum(tp.probabilidad * tp.tiempo_servicio for tp in tipos)
    cpu_media = sum(tp.probabilidad * (tp.cpu_min + tp.cpu_max) / 2 for tp in tipos)
    tasa_observada = m.generadas / t
    colas = {
        "tasa_llegada_observada": tasa_observada,
        "tiempo_servicio_medio": servicio_medio,
        "servidores_equivalentes": int(sim.capacidad_total // cpu_media),
    }
    if tasa_observada > 0:
        mmc = calcular_mmc(tasa_observada, 1 / servicio_medio, colas["servidores_equivalentes"])
        colas["mmc"] = mmc.__dict__
        colas["little"] = ley_de_little(m.promedio("en_sistema"), m.procesadas / t, m.tiempo_sistema_promedio)

    # --- Módulo 1: rutas y mochila ------------------------------------------
    rutas = sim.resultado_rutas
    ids_servidores = {s.id for s in sim.servidores}
    tabla = [  # tabla de la recursión (sin la última etapa: el salto ToR → servidor es fijo)
        {
            "etapa": fila.etapa,
            "nodo": fila.nodo,
            "candidatos": ", ".join(f"{c.desde}: {c.costo:.2f}" for c in fila.candidatos) or "-",
            "f": fila.valor_optimo,
            "decision": fila.decision or "-",
        }
        for fila in rutas.tabla
        if fila.nodo not in ids_servidores
    ]
    motivos: dict[str, int] = {}
    for cambio in sim.cambios_ruta:
        motivos[cambio.motivo] = motivos.get(cambio.motivo, 0) + 1

    decision = sim.mochila_destacada or sim.ultima_mochila
    mochila = {
        "ejecuciones": m.ejecuciones_mochila,
        "utilizacion_promedio": m.utilizacion_mochila_promedio,
        "destacada": None,
    }
    if decision:
        mochila["destacada"] = {
            "t": decision.t,
            "servidor": decision.servidor,
            "capacidad": decision.capacidad,
            "valor_total": decision.valor_total,
            "peso_total": decision.peso_total,
            "articulos": [
                {"tarea": a.nombre, "peso": a.peso, "valor": a.valor, "elegida": i in decision.seleccionados}
                for i, a in enumerate(decision.articulos[:25])
            ],
            "total_articulos": len(decision.articulos),
        }

    # --- Módulo 2 -----------------------------------------------------------
    analisis = sim.analisis_termico
    gradiente, lagrange = sim.ultimo_gradiente, sim.ultimo_lagrange
    verificacion = sim.asignador.verificar(lagrange, sim.ultimos_coeficientes)
    hessiano = sim.asignador.hessiano_orlado(lagrange, sim.ultimos_coeficientes)
    verificacion["menores_hessiano_orlado"] = hessiano["menores"]
    verificacion["es_maximo"] = hessiano["es_maximo"]

    return {
        "generado": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parametros": {
            "tiempo_simulado": sim.ahora,
            "duracion_configurada": cfg.simulacion.duracion,
            "semilla": cfg.simulacion.semilla,
            "tasa_llegada_base": sim.tasa_llegada,
            "amplitud_pico": cfg.simulacion.amplitud_pico,
            "periodo_pico": cfg.simulacion.periodo_pico,
            "presupuesto_electrico": sim.presupuesto,
            "capacidad_total": sim.capacidad_total,
            "servidores": len(sim.servidores),
            "clusters": nombres,
        },
        "indicadores": {
            "generadas": m.generadas,
            "procesadas": m.procesadas,
            "descartadas": m.descartadas,
            "retransmitidas": m.retransmitidas,
            "interrumpidas": m.interrumpidas,
            "en_cola_final": len(sim.cola),
            "porcentaje_descarte": m.descartadas / m.generadas if m.generadas else 0.0,
            "throughput_promedio": m.procesadas / t,
            "espera_promedio": m.espera_promedio,
            "tiempo_sistema_promedio": m.tiempo_sistema_promedio,
            "utilizacion_promedio": m.promedio("utilizacion"),
            "cola_promedio": m.promedio("en_cola"),
            "valor_procesado": m.valor_procesado,
            "fallas": m.fallas,
            "cambios_ruta": m.cambios_ruta,
            "completadas_por_tipo": dict(m.completadas_por_tipo),
            "descartadas_por_tipo": dict(m.descartadas_por_tipo),
        },
        "colas": colas,
        "rutas": {
            "algoritmo": cfg.algoritmo_rutas,
            "tabla": tabla,
            "por_cluster": [
                {
                    "cluster": c.nombre,
                    "ruta": _ruta(sim.rutas_actuales.get(c.nombre)),  # ruta en uso
                    "latencia": sim.latencias_actuales.get(c.nombre),
                    "ruta_optima": _ruta(sim.rutas_optimas.get(c.nombre)),  # resultado de la DP
                    "latencia_optima": sim.latencias_optimas.get(c.nombre),
                }
                for c in sim.clusters
            ],
            "umbral_cambio_ruta": UMBRAL_CAMBIO_RUTA,
            "cambios_por_motivo": motivos,
            "ultimos_cambios": [
                {"t": c.t, "cluster": c.cluster, "anterior": _ruta(c.anterior), "nueva": _ruta(c.nueva), "motivo": c.motivo}
                for c in sim.cambios_ruta[-8:]
            ],
        },
        "mochila": mochila,
        "termico": {
            "funcion": analisis.funcion,
            "dominio": list(sim.modelo_termico.dominio),
            "primera_derivada": analisis.primera_derivada,
            "segunda_derivada": analisis.segunda_derivada,
            "estacionarios": [p.__dict__ for p in analisis.puntos_estacionarios],
            "inflexiones": [p.__dict__ for p in analisis.puntos_inflexion],
            "temperatura_emergencia": analisis.temperatura_emergencia,
            "carga_emergencia": analisis.carga_emergencia,
            "zona_estable_hasta": analisis.zona_estable_hasta,
            "maximo_absoluto": analisis.maximo_absoluto.__dict__,
            "temperatura_maxima_observada": max((max(x.temperaturas) for x in m.muestras), default=0.0),
            "eventos_emergencia": sum(1 for e in m.eventos if e.tipo == "emergencia"),
        },
        "refrigeracion": {
            "niveles": dict(zip(nombres, gradiente.niveles)),
            "energia_actual": gradiente.energia,
            "iteraciones_ultimo_ciclo": gradiente.iteraciones,
            "convergio": gradiente.convergio,
            "energia_promedio": m.promedio("energia_refrigeracion"),
            "iteraciones_promedio": m.promedio("iteraciones_gradiente"),
            "tasa_aprendizaje": cfg.refrigeracion.tasa_aprendizaje,
        },
        "energia": {
            "metodo": lagrange.metodo,
            "presupuesto": lagrange.presupuesto,
            "multiplicador": lagrange.multiplicador,
            "multiplicador_promedio": m.promedio("multiplicador"),
            "rendimiento_total": lagrange.rendimiento_total,
            "asignacion": [
                {"cluster": c.nombre, "coeficiente": a, "energia": e, "rendimiento": r, "factor_velocidad": c.factor_velocidad}
                for c, a, e, r in zip(sim.clusters, sim.ultimos_coeficientes, lagrange.energias, lagrange.rendimientos)
            ],
            "verificacion": verificacion,
        },
        # Estado de cada clúster AL FINAL de la simulación (los máximos están en "termico").
        "clusters_al_final": [
            {
                "nombre": c.nombre,
                "servidores_activos_al_final": f"{c.servidores_activos}/{len(c.servidores)}",
                "carga_al_final": c.carga,
                "temperatura_al_final": c.temperatura_max,
                "emergencia_al_final": c.en_emergencia,
                "tareas_completadas": sum(s.completadas for s in c.servidores),
            }
            for c in sim.clusters
        ],
        "eventos": [{"t": e.t, "tipo": e.tipo, "mensaje": e.mensaje} for e in list(m.eventos)[-30:]],
        # Series para las gráficas del PDF (no se envían a la IA)
        "series": {
            "t": [x.t for x in m.muestras],
            "en_cola": [x.en_cola for x in m.muestras],
            "throughput": [x.throughput for x in m.muestras],
            "utilizacion": [x.utilizacion for x in m.muestras],
            "tasa_llegada": [x.tasa_llegada for x in m.muestras],
            "niveles_ventilador": [list(x.niveles_ventilador) for x in m.muestras],
            "energia_refrigeracion": [x.energia_refrigeracion for x in m.muestras],
            "energias_cluster": [list(x.energias_cluster) for x in m.muestras],
            "multiplicador": [x.multiplicador for x in m.muestras],
            "temperaturas": [list(x.temperaturas) for x in m.muestras],
            "historial_gradiente": list(gradiente.historial_energia),
        },
    }
