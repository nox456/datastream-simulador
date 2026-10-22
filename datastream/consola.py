"""Ejecución independiente de cada módulo con salida por consola."""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Sequence

from .configuracion import Configuracion
from .modelos.bellman_ford import BellmanFordDP
from .modelos.mochila import Articulo, ModeloMochilaUnidades, ModeloVolumenCarga, ResultadoMochilaUnidades
from .modelos.presupuesto import AsignadorEnergia
from .modelos.programacion_lineal import (
    ProblemaLineal,
    ResolvedorSimplex,
    Restriccion,
    ResultadoSimplex,
    TablaSimplex,
    resolver_con_scipy,
)
from .modelos.refrigeracion import OptimizadorRefrigeracion
from .modelos.ruta_mas_corta import INFINITO, ResultadoHaciaAtras, ResultadoRutas, RutaMasCortaDP
from .modelos.temperatura import ModeloTermico


def imprimir_tabla(encabezados: Sequence[str], filas: Sequence[Sequence]) -> None:
    textos = [[str(c) for c in fila] for fila in filas]
    anchos = [max(len(str(h)), *(len(f[i]) for f in textos)) if textos else len(str(h)) for i, h in enumerate(encabezados)]
    separador = "+-" + "-+-".join("-" * a for a in anchos) + "-+"
    print(separador)
    print("| " + " | ".join(str(h).ljust(a) for h, a in zip(encabezados, anchos)) + " |")
    print(separador)
    for fila in textos:
        print("| " + " | ".join(c.ljust(a) for c, a in zip(fila, anchos)) + " |")
    print(separador)


def titulo(texto: str) -> None:
    print(f"\n{'=' * 72}\n {texto}\n{'=' * 72}")


# ---------------------------------------------------------------------- #
# Módulo 1
# ---------------------------------------------------------------------- #
def _numero(valor: float) -> str:
    return f"{valor:.2f}" if valor != INFINITO else "inf"


def _imprimir_hacia_adelante(resultado: ResultadoRutas, etiqueta_etapa: str = "Etapa") -> None:
    filas = [
        [
            f.etapa,
            f.nodo,
            ", ".join(f"{c.desde}: {c.costo:.2f}" for c in f.candidatos) or "sin enlaces",
            _numero(f.valor_optimo),
            f.decision or "-",
        ]
        for f in resultado.tabla
    ]
    imprimir_tabla([etiqueta_etapa, "Nodo y", "Candidatos x: f(x)+d(x,y)", "f_k(y)", "Decisión x*"], filas)


def _imprimir_hacia_atras(resultado: ResultadoHaciaAtras) -> None:
    """Una tabla por etapa, empezando por la última, con el formato visto en clase."""
    etapas = sorted({f.etapa for f in resultado.tabla}, reverse=True)
    for etapa in etapas:
        filas = [f for f in resultado.tabla if f.etapa == etapa]
        print(f"\nEtapa {etapa}:  f_{etapa}(x_{etapa}) = min {{ d(x_{etapa}, x_{etapa + 1}) + f_{etapa + 1}(x_{etapa + 1}) }}")
        imprimir_tabla(
            [f"x_{etapa}", f"Candidatos x_{etapa + 1}: d + f_{etapa + 1}", f"f_{etapa}(x_{etapa})", f"x_{etapa + 1}*"],
            [
                [
                    f.nodo,
                    ", ".join(f"{c.hacia}: {c.costo:.2f}" for c in f.candidatos) or "sin enlaces",
                    _numero(f.valor_optimo),
                    f.decision or "-",
                ]
                for f in filas
            ],
        )


def _arcos_activos(config: Configuracion, nodos_caidos: Sequence[str]) -> dict[tuple[str, str], float]:
    caidos = set(nodos_caidos)
    if caidos:
        print(f"Nodos fuera de servicio: {', '.join(sorted(caidos))}")
    return {
        (e.origen, e.destino): e.latencia
        for e in config.enlaces
        if e.origen not in caidos and e.destino not in caidos
    }


def _imprimir_politica(config: Configuracion, resultado: ResultadoRutas) -> None:
    print("\nPolítica óptima (ruta reconstruida desde cada destino):")
    imprimir_tabla(
        ["Clúster", "Switch", "Ruta", "Latencia (s)"],
        [
            [
                c.nombre,
                c.switch,
                " > ".join(resultado.ruta_hacia(c.switch) or ["inalcanzable"]),
                _numero(resultado.latencia_hacia(c.switch)),
            ]
            for c in config.clusters
        ],
    )


def demo_ruta(config: Configuracion, nodos_caidos: Sequence[str] = (), destino: str | None = None) -> None:
    if config.algoritmo_rutas == "bellman_ford":
        demo_bellman_ford(config, nodos_caidos)
        return
    titulo("MÓDULO 1 · Ruta más corta con programación dinámica")
    etapas = [list(e) for e in config.etapas]
    arcos = _arcos_activos(config, nodos_caidos)
    modelo = RutaMasCortaDP(etapas)

    print("\n1) Recursión hacia adelante (desde el gateway hacia todos los nodos, latencias base en s):")
    print("   f_0(GW) = 0 ;  f_k(y) = min_x { f_(k-1)(x) + d(x, y) }\n")
    resultado = modelo.resolver(arcos)
    _imprimir_hacia_adelante(resultado)

    destino = destino or config.clusters[0].switch
    print(f"\n2) Recursión hacia atrás (desde el destino {destino} hasta el gateway):")
    print(f"   f_n({destino}) = 0 ;  f_i(x_i) = min {{ d(x_i, x_(i+1)) + f_(i+1)(x_(i+1)) }}")
    atras = modelo.resolver_hacia_atras(arcos, destino)
    _imprimir_hacia_atras(atras)
    ruta_atras = atras.ruta()
    ruta_adelante = resultado.ruta_hacia(destino)
    print(f"\nRuta hacia atrás:    {' > '.join(ruta_atras or ['inalcanzable'])}  ·  f = {_numero(atras.latencia)}")
    print(f"Ruta hacia adelante: {' > '.join(ruta_adelante or ['inalcanzable'])}  ·  f = {_numero(resultado.latencia_hacia(destino))}")
    if ruta_atras and ruta_adelante:
        iguales = math.isclose(atras.latencia, resultado.latencia_hacia(destino))
        print("Ambas recursiones dan la misma latencia óptima." if iguales else "ATENCIÓN: las recursiones no coinciden.")

    _imprimir_politica(config, resultado)


def demo_bellman_ford(config: Configuracion, nodos_caidos: Sequence[str] = ()) -> None:
    titulo("MÓDULO 1 · Ruta más corta con Bellman-Ford (programación dinámica sobre un grafo general)")
    arcos = _arcos_activos(config, nodos_caidos)
    nodos = [n for etapa in config.etapas for n in etapa]
    resultado = BellmanFordDP(nodos, config.etapas[0][0]).resolver(arcos)
    laterales = [f"{u}→{v}" for u, v in arcos if _etapa(config, v) != _etapa(config, u) + 1]
    print("f_0(GW) = 0 ;  f_k(v) = min { f_(k-1)(v), min_u [ f_(k-1)(u) + d(u, v) ] }   (k = enlaces usados)")
    if laterales:
        print(f"Enlaces que la DP por etapas no admite: {', '.join(laterales)}")

    print("\n1) Iteraciones: f_k(v) es la latencia mínima usando a lo sumo k enlaces")
    columnas = range(len(resultado.iteraciones))
    imprimir_tabla(
        ["Nodo v"] + [f"f_{k}" for k in columnas],
        [[v] + [_numero(resultado.iteraciones[k][v]) for k in columnas] for v in nodos],
    )
    print(f"Se detuvo en k = {len(resultado.iteraciones) - 1}: en la siguiente iteración ningún valor mejoró.")
    if resultado.ciclo_negativo:
        print("ATENCIÓN: hay un ciclo de latencia negativa; la ruta más corta no está definida.")
        return

    print("\n2) Tabla final (k = número de enlaces de la ruta óptima):")
    _imprimir_hacia_adelante(resultado, "Enlaces k")
    _imprimir_politica(config, resultado)

    if not laterales:
        por_etapas = RutaMasCortaDP([list(e) for e in config.etapas]).resolver(arcos)
        iguales = all(
            math.isclose(por_etapas.latencia_hacia(c.switch), resultado.latencia_hacia(c.switch)) for c in config.clusters
        )
        print("La DP por etapas da las mismas latencias óptimas." if iguales
              else "ATENCIÓN: la DP por etapas da latencias distintas.")


def _etapa(config: Configuracion, nodo: str) -> int:
    return next(k for k, etapa in enumerate(config.etapas) if nodo in etapa)


# Datos de los ejemplos resueltos en la Lectura 1 de Programación Dinámica (láminas 4-13 y 18-21).
ETAPAS_CLASE = [["1"], ["2", "3", "4"], ["5", "6"], ["7"]]
ARCOS_CLASE = {
    ("1", "2"): 7, ("1", "3"): 8, ("1", "4"): 5,
    ("2", "5"): 12, ("3", "5"): 8, ("3", "6"): 9, ("4", "5"): 7, ("4", "6"): 13,
    ("5", "7"): 9, ("6", "7"): 6,
}
ARTICULOS_BARCO = [Articulo("Artículo 1", 2, 31), Articulo("Artículo 2", 3, 47), Articulo("Artículo 3", 1, 14)]
CAPACIDAD_BARCO = 4


def demo_ruta_clase() -> None:
    titulo("EJEMPLO DE CLASE · Ruta más corta entre las ciudades 1 y 7 (respuesta esperada: 1 > 4 > 5 > 7, 21 km)")
    modelo = RutaMasCortaDP(ETAPAS_CLASE)
    print("Recursión hacia adelante:")
    adelante = modelo.resolver(ARCOS_CLASE)
    _imprimir_hacia_adelante(adelante)
    print("\nRecursión hacia atrás:")
    atras = modelo.resolver_hacia_atras(ARCOS_CLASE, "7")
    _imprimir_hacia_atras(atras)
    print(f"\nRuta más corta: {' > '.join(atras.ruta())} con {atras.latencia:.0f} km "
          f"(hacia adelante: {' > '.join(adelante.ruta_hacia('7'))}, {adelante.latencia_hacia('7'):.0f} km)")
    bellman = BellmanFordDP([n for etapa in ETAPAS_CLASE for n in etapa], "1").resolver(ARCOS_CLASE)
    print(f"Con Bellman-Ford (grafo general): {' > '.join(bellman.ruta_hacia('7'))} con "
          f"{bellman.latencia_hacia('7'):.0f} km en {len(bellman.iteraciones) - 1} iteraciones")


def demo_mochila_clase() -> None:
    titulo("EJEMPLO DE CLASE · Barco de 4 toneladas (respuesta esperada: 2 unidades del artículo 1, $62.000)")
    imprimir_tabla(
        ["Artículo i", "Peso unitario w_i (t)", "Ingreso unitario r_i (miles $)"],
        [[a.nombre, a.peso, f"{a.valor:.0f}"] for a in ARTICULOS_BARCO],
    )
    resultado = ModeloMochilaUnidades().resolver(ARTICULOS_BARCO, CAPACIDAD_BARCO)
    _imprimir_mochila_unidades(resultado)


def _imprimir_mochila_unidades(resultado: ResultadoMochilaUnidades) -> None:
    """Tablas f_i(x_i) de la recursión hacia atrás, de la última etapa a la primera."""
    n = len(resultado.etapas)
    for calculadas, etapa in enumerate(resultado.etapas):  # se calcularon del último artículo al primero
        i = n - calculadas
        a = etapa.articulo
        siguiente = f" + f_{i + 1}(x_{i} - {a.peso}·m_{i})" if i < n else ""
        print(f"\nEtapa {i} ({a.nombre}):  f_{i}(x_{i}) = max {{ {a.valor:g}·m_{i}{siguiente} }},  "
              f"m_{i} = 0, …, ⌊{resultado.capacidad}/{a.peso}⌋ = {etapa.maximo_unidades}")
        imprimir_tabla(
            [f"x_{i}"] + [f"m_{i}={m}" for m in range(etapa.maximo_unidades + 1)] + [f"f_{i}(x_{i})", f"m_{i}*"],
            [
                [fila.estado]
                + ["-" if v is None else f"{v:g}" for v in fila.alternativas]
                + [f"{fila.valor_optimo:g}", fila.decision]
                for fila in etapa.filas
            ],
        )
    detalle = ", ".join(f"{m} de {a.nombre}" for a, m in zip((e.articulo for e in reversed(resultado.etapas)), resultado.cantidades))
    print(f"\nSolución óptima: {detalle} · rendimiento {resultado.valor_total:g} · "
          f"peso {resultado.peso_total}/{resultado.capacidad}")


def demo_mochila(config: Configuracion, capacidad: int, cantidad: int, semilla: int) -> None:
    titulo("MÓDULO 1 · Modelo de volumen-carga (mochila 0/1)")
    azar = random.Random(semilla)
    tipos = config.tipos_tarea
    articulos = []
    for i in range(1, cantidad + 1):
        tipo = azar.choices(tipos, weights=[t.probabilidad for t in tipos])[0]
        cpu = azar.randint(tipo.cpu_min, tipo.cpu_max)
        prioridad = azar.randint(tipo.prioridad_min, tipo.prioridad_max)
        articulos.append((Articulo(f"T{i}", cpu, float(prioridad * cpu)), tipo.nombre, prioridad))

    print(f"Capacidad del servidor W = {capacidad} unidades de CPU · {cantidad} tareas en cola (semilla {semilla})")
    print("Valor de cada tarea v_i = prioridad · cpu\n")
    resultado = ModeloVolumenCarga().resolver([a for a, _, _ in articulos], capacidad, guardar_tabla=True)
    imprimir_tabla(
        ["Tarea", "Tipo", "Prioridad", "Peso w_i", "Valor v_i", "m_i"],
        [
            [a.nombre, tipo, prioridad, a.peso, f"{a.valor:.0f}", 1 if i in resultado.seleccionados else 0]
            for i, (a, tipo, prioridad) in enumerate(articulos)
        ],
    )
    if capacidad <= 16 and cantidad <= 10:
        print("\nTabla f_i(w) (filas: tareas consideradas, columnas: capacidad w):")
        imprimir_tabla(
            ["i \\ w"] + [str(w) for w in range(capacidad + 1)],
            [[i] + [f"{v:.0f}" for v in fila] for i, fila in enumerate(resultado.tabla)],
        )
    print(
        f"\nSolución óptima: f_{cantidad}({capacidad}) = {resultado.valor_total:.0f} · "
        f"peso usado {resultado.peso_total}/{capacidad} ({resultado.utilizacion:.0%}) · "
        f"tareas elegidas: {', '.join(articulos[i][0].nombre for i in resultado.seleccionados) or 'ninguna'}"
    )


# ---------------------------------------------------------------------- #
# Módulo 2
# ---------------------------------------------------------------------- #
def demo_temperatura(config: Configuracion, temperatura_emergencia: float | None = None) -> None:
    titulo("MÓDULO 2 · Análisis térmico T(x) (modelo no lineal de una variable)")
    t = config.termico
    umbral = t.temperatura_emergencia if temperatura_emergencia is None else temperatura_emergencia
    modelo = ModeloTermico(t.funcion, (t.carga_min, t.carga_max), umbral)
    analisis = modelo.analizar()
    print(f"T(x)   = {analisis.funcion}")
    print(f"T'(x)  = {analisis.primera_derivada}")
    print(f"T''(x) = {analisis.segunda_derivada}\n")
    filas = [
        [p.tipo, f"{p.carga:.2f}", f"{p.temperatura:.2f}", "T'(x)=0"] for p in analisis.puntos_estacionarios
    ] + [[p.tipo, f"{p.carga:.2f}", f"{p.temperatura:.2f}", "T''(x)=0"] for p in analisis.puntos_inflexion]
    imprimir_tabla(["Punto", "Carga x (%)", "T (°C)", "Condición"], filas or [["sin puntos notables", "-", "-", "-"]])
    print(f"\nMáximo absoluto en el dominio: {analisis.maximo_absoluto.temperatura:.2f} °C en x = {analisis.maximo_absoluto.carga:.1f}%")
    if analisis.zona_estable_hasta is not None:
        print(f"Zona térmicamente estable hasta el mínimo local: x = {analisis.zona_estable_hasta:.1f}%")
    if analisis.carga_emergencia is not None:
        print(f"PUNTO CRÍTICO: con x ≥ {analisis.carga_emergencia:.2f}% se alcanza T ≥ {umbral:.1f} °C "
              "→ se requiere enfriamiento de emergencia.")
    else:
        print(f"El servidor nunca alcanza {umbral:.1f} °C dentro del dominio de carga.")
    print("\nTabla de valores:")
    paso = (t.carga_max - t.carga_min) / 10
    cargas = [t.carga_min + i * paso for i in range(11)]
    imprimir_tabla(["x (%)"] + [f"{c:.0f}" for c in cargas], [["T (°C)"] + [f"{modelo.temperatura(c):.1f}" for c in cargas]])


def demo_gradiente(config: Configuracion, cargas: Sequence[float]) -> None:
    titulo("MÓDULO 2 · Método de gradiente para la refrigeración")
    r = config.refrigeracion
    optimizador = OptimizadorRefrigeracion(
        r.coef_ventilador, r.coef_calor, r.coef_acople,
        vecinos=[(i, i + 1) for i in range(len(cargas) - 1)],
        tasa_aprendizaje=r.tasa_aprendizaje, tolerancia=r.tolerancia,
        max_iteraciones=r.max_iteraciones, nivel_minimo=r.nivel_minimo,
    )
    print(f"E(u) = Σ [{r.coef_ventilador}·u_i³ + {r.coef_calor}·L_i²/u_i] + {r.coef_acople}·Σ (u_i - u_j)²")
    print("∂E/∂u_i = 3A·u_i² - B·L_i²/u_i² + 2C·Σ_j (u_i - u_j)")
    print(f"u^(k+1) = Proy[ u^(k) - α·∇E(u^(k)) ],  α = {r.tasa_aprendizaje}\n")
    print("Cargas L_i: " + ", ".join(f"{c.nombre}={l:.2f}" for c, l in zip(config.clusters, cargas)))

    resultado = optimizador.optimizar(cargas, niveles_iniciales=[0.5] * len(cargas))
    iteraciones = {0, 1, 2, 3, 5, 10, 20, 40, 80, 160, resultado.iteraciones}
    filas = []
    for k in sorted(i for i in iteraciones if i <= resultado.iteraciones):
        niveles = resultado.historial_niveles[k]
        gradiente = optimizador.gradiente(niveles, cargas)
        filas.append([k, " ".join(f"{v:.3f}" for v in niveles), f"{resultado.historial_energia[k]:.4f}",
                      f"{max(abs(g) for g in gradiente):.4f}"])
    imprimir_tabla(["k", "u^(k)", "E(u^(k)) kW", "max |∂E/∂u_i|"], filas)
    estado = "convergió" if resultado.convergio else "alcanzó el máximo de iteraciones"
    print(f"\nResultado: {estado} en {resultado.iteraciones} iteraciones · E* = {resultado.energia:.4f} kW")
    imprimir_tabla(
        ["Clúster", "Carga L_i", "Ventilador u_i*"],
        [[c.nombre, f"{l:.2f}", f"{u_i:.1%}"] for c, l, u_i in zip(config.clusters, cargas, resultado.niveles)],
    )


def demo_lagrange(
    config: Configuracion, presupuesto: float, coeficientes: Sequence[float], metodo: str = "analitico"
) -> None:
    titulo("MÓDULO 2 · Asignación del presupuesto eléctrico con multiplicadores de Lagrange")
    asignador = AsignadorEnergia(presupuesto, metodo)
    simbolico = asignador.derivacion_simbolica(len(coeficientes))
    print(f"L(E, λ) = {simbolico['lagrangiano']}\n")
    print("Condiciones de primer orden (derivadas parciales = 0):")
    for condicion in simbolico["condiciones"]:
        print(f"   {condicion.lhs} = 0")
    if metodo == "simbolico":
        print("\nsp.solve resolvió el sistema con coeficientes simbólicos:")
        for incognita, expresion in asignador.solucion_simbolica(len(coeficientes)).items():
            print(f"   {incognita}* = {expresion}")
        print()
    elif metodo == "numerico":
        print("\nSciPy (SLSQP) maximiza R(E) directamente con la restricción Σ E_i = E_max, sin usar la fórmula.\n")
    else:
        print("\n⇒ E_i* = E_max·a_i² / Σa_j²   y   λ* = √(Σa_i²) / (2·√E_max)\n")
    resultado = asignador.asignar(coeficientes)
    imprimir_tabla(
        ["Clúster", "a_i", "E_i* (kW)", "% del total", "R_i = a_i·√E_i"],
        [
            [c.nombre, f"{a:.3f}", f"{e:.2f}", f"{e / presupuesto:.1%}", f"{r:.4f}"]
            for c, a, e, r in zip(config.clusters, coeficientes, resultado.energias, resultado.rendimientos)
        ],
    )
    verificacion = asignador.verificar(resultado, coeficientes)
    print(f"\nλ* = {resultado.multiplicador:.6f} (rendimiento adicional por cada kW extra)")
    print(f"Rendimiento total R* = {resultado.rendimiento_total:.4f} · Σ E_i = {sum(resultado.energias):.2f} kW")
    print(f"Verificación: max |∂L/∂E_i| = {verificacion['max_residuo_dL_dEi']:.2e} · "
          f"restricción = {verificacion['residuo_restriccion']:.2e}")
    if metodo != "analitico":
        analitico = AsignadorEnergia(presupuesto).asignar(coeficientes)
        diferencia = max(abs(x - y) for x, y in zip(resultado.energias, analitico.energias))
        print(f"Comparado con la fórmula analítica: diferencia máxima en E_i* = {diferencia:.2e} kW, "
              f"en λ* = {abs(resultado.multiplicador - analitico.multiplicador):.2e}")
    print("Como ∂²R/∂E_i² = -a_i/(4·E_i^1.5) < 0, la función es cóncava y el punto es un máximo.")

    hessiano = asignador.hessiano_orlado(resultado, coeficientes)
    if hessiano["menores"]:
        print("\nCondición de segundo orden · Hessiano orlado H = [[0, ∇g], [∇g, ∂²L/∂E_i∂E_j]]:")
        for fila in hessiano["matriz"]:
            print("   [" + "  ".join(f"{v:>11.3e}" for v in fila) + " ]")
        imprimir_tabla(
            ["Orden del menor", "Valor", "Signo requerido para máximo"],
            [
                [orden, f"{m:+.3e}", "+" if orden % 2 == 1 else "−"]
                for orden, m in zip(range(3, len(hessiano["menores"]) + 3), hessiano["menores"])
            ],
        )
        print("Los signos alternan (+, −, +, …): el punto es un MÁXIMO." if hessiano["es_maximo"]
              else "Los signos NO alternan como se requiere: el punto no es un máximo.")


def coeficientes_por_defecto(config: Configuracion, carga: float = 0.5) -> list[float]:
    """a_i que usaría la simulación con todos los servidores activos y carga uniforme."""
    capacidad_total = sum(s.capacidad for c in config.clusters for s in c.servidores)
    return [
        c.eficiencia * (sum(s.capacidad for s in c.servidores) / capacidad_total) * (config.energia.carga_base + carga)
        for c in config.clusters
    ]


MINIMO_WEB = 10
EJEMPLOS_LINEAL = ("centro-datos", "parcial-iv", "minimizar", "no-acotado", "infactible")


def problema_centro_datos(config: Configuracion) -> ProblemaLineal:
    """Tareas simultáneas de cada tipo que maximizan el rendimiento sin exceder la CPU ni la energía.

    Para cada tipo j, cpu_j y prioridad_j son los puntos medios de sus rangos, el rendimiento
    es r_j = prioridad_j·cpu_j (el mismo valor de la mochila) y el consumo e_j = 0.1·cpu_j·t_j kW
    (las tareas pesadas y largas consumen más). La CPU disponible es la capacidad total de los
    servidores, la energía es el presupuesto eléctrico y se exige un mínimo de tareas web.
    """
    tipos = config.tipos_tarea
    cpu = [(t.cpu_min + t.cpu_max) / 2 for t in tipos]
    rendimiento = [(t.prioridad_min + t.prioridad_max) / 2 * c for t, c in zip(tipos, cpu)]
    consumo = [0.1 * c * t.tiempo_servicio for t, c in zip(tipos, cpu)]
    restricciones = [
        Restriccion(tuple(cpu), "<=", float(sum(s.capacidad for c in config.clusters for s in c.servidores))),
        Restriccion(tuple(consumo), "<=", config.energia.presupuesto_total),
    ]
    restricciones += [
        Restriccion(tuple(1.0 if j == i else 0.0 for j in range(len(tipos))), ">=", MINIMO_WEB)
        for i, t in enumerate(tipos)
        if t.nombre == "web"
    ]
    return ProblemaLineal(tuple(rendimiento), tuple(restricciones), "max", tuple(t.nombre for t in tipos))


def problema_ejemplo(nombre: str, config: Configuracion) -> tuple[ProblemaLineal, str]:
    if nombre == "centro-datos":
        return problema_centro_datos(config), "Centro de datos: tareas por tipo con CPU y energía limitadas"
    if nombre == "parcial-iv":
        return ProblemaLineal(
            (100, 40), (Restriccion((2, 1), "<=", 9), Restriccion((10, 4), "<=", 15))
        ), "Relajación lineal del problema 2 del Parcial IV (soluciones múltiples)"
    if nombre == "minimizar":
        return ProblemaLineal(
            (4, 1),
            (Restriccion((3, 1), "=", 3), Restriccion((4, 3), ">=", 6), Restriccion((1, 2), "<=", 4)),
            "min",
        ), "Minimización con restricciones =, >= y <= (dos fases)"
    if nombre == "no-acotado":
        return ProblemaLineal(
            (2, 1), (Restriccion((1, -1), "<=", 10), Restriccion((2, 0), "<=", 40))
        ), "Solución no acotada"
    if nombre == "infactible":
        return ProblemaLineal(
            (3, 2), (Restriccion((2, 1), "<=", 2), Restriccion((3, 4), ">=", 12))
        ), "Problema infactible"
    raise ValueError(f"Ejemplo desconocido '{nombre}'. Opciones: {', '.join(EJEMPLOS_LINEAL)}")


def problemas_lineales(config: Configuracion, propio: ProblemaLineal | None = None) -> dict[str, tuple[ProblemaLineal, str]]:
    """Todos los ejemplos (y el problema propio, si hay) para elegirlos en la interfaz gráfica."""
    problemas = {nombre: problema_ejemplo(nombre, config) for nombre in EJEMPLOS_LINEAL}
    if propio is not None:
        problemas["propio"] = (propio, "Problema ingresado")
    return problemas


def demo_lineal_ejemplo(nombre: str, config: Configuracion) -> ResultadoSimplex:
    problema, descripcion = problema_ejemplo(nombre, config)
    datos = None
    if nombre == "centro-datos":
        datos = (
            ["Tipo j", "cpu_j", "prioridad_j", "r_j = prioridad·cpu", "t_j (s)", "e_j = 0.1·cpu·t (kW)"],
            [
                [t.nombre, f"{(t.cpu_min + t.cpu_max) / 2:g}", f"{(t.prioridad_min + t.prioridad_max) / 2:g}",
                 f"{r:g}", f"{t.tiempo_servicio:g}", f"{e:g}"]
                for t, r, e in zip(config.tipos_tarea, problema.objetivo, problema.restricciones[1].coeficientes)
            ],
        )
    return demo_lineal(problema, descripcion, datos)


def demo_lineal(
    problema: ProblemaLineal,
    descripcion: str = "Problema ingresado",
    datos: tuple[Sequence[str], Sequence[Sequence]] | None = None,
) -> ResultadoSimplex:
    titulo(f"PROGRAMACIÓN LINEAL · Método Simplex · {descripcion}")
    if datos is not None:
        imprimir_tabla(*datos)
        print()
    nombres = problema.nombres
    print(f"{problema.sentido} Z = {expresion_lineal(problema.objetivo, nombres)}")
    print("s.a.")
    for i, r in enumerate(problema.restricciones, start=1):
        print(f"   ({i}) {expresion_lineal(r.coeficientes, nombres)} {r.sentido} {r.lado_derecho:g}")
    print(f"   {', '.join(nombres)} >= 0")

    resultado = ResolvedorSimplex().resolver(problema)
    _imprimir_forma_estandar(problema, resultado)
    for tabla in resultado.tablas:
        _imprimir_tabla_simplex(tabla, resultado.usa_dos_fases)
    _imprimir_resultado_lineal(problema, resultado)
    return resultado


def expresion_lineal(coeficientes: Sequence[float], nombres: Sequence[str]) -> str:
    terminos = []
    for a, x in zip(coeficientes, nombres):
        if a == 0:
            continue
        magnitud = "" if abs(a) == 1 else f"{abs(a):g}·"
        signo = ("-" if a < 0 else "") if not terminos else (" - " if a < 0 else " + ")
        terminos.append(f"{signo}{magnitud}{x}")
    return "".join(terminos) or "0"


def fraccion(valor: float) -> str:
    """Muestra el número como fracción (2/5, -7/3) cuando lo es, igual que en las tablas hechas a mano."""
    aproximada = Fraction(valor).limit_denominator(1000)
    if abs(float(aproximada) - valor) > 1e-9:
        return f"{valor:.4g}"
    if aproximada.denominator == 1:
        return str(aproximada.numerator)
    return f"{aproximada.numerator}/{aproximada.denominator}"


def _imprimir_forma_estandar(problema: ProblemaLineal, resultado: ResultadoSimplex) -> None:
    agregadas = resultado.tablas[0].columnas[len(problema.objetivo):]
    print("\nForma estándar:")
    if problema.sentido == "min":
        print("   min Z equivale a max (-Z): se maximiza -Z y al final se cambia el signo.")
    if any(r.lado_derecho < 0 for r in problema.restricciones):
        print("   Las restricciones con b_i < 0 se multiplicaron por -1 (cambia su sentido).")
    descripciones = {
        "s": "Variables de holgura (<=)",
        "e": "Variables de exceso (>=, coeficiente -1)",
        "a": "Variables artificiales (>= o =)",
    }
    for prefijo, texto in descripciones.items():
        variables = [v for v in agregadas if v.startswith(prefijo)]
        if variables:
            print(f"   {texto}: {', '.join(variables)}")
    if resultado.usa_dos_fases:
        artificiales = [v for v in agregadas if v.startswith("a")]
        print(f"   Fase 1: max W = -({' + '.join(artificiales)}) para llevar las artificiales a 0.")
        print("   Fase 2: se eliminan las artificiales y se optimiza Z desde la base factible obtenida.")
    print("   Fila objetivo: z_j - c_j. Entra la más negativa; sale la de menor razón b_i / a_ik (a_ik > 0).")


def _imprimir_tabla_simplex(tabla: TablaSimplex, dos_fases: bool, encabezado: str | None = None) -> None:
    print(f"\n{encabezado or (f'Fase {tabla.fase} · ' if dos_fases else '') + f'Tabla {tabla.iteracion}'}")
    con_razones = bool(tabla.razones)
    filas = []
    for i, (variable, fila) in enumerate(zip(tabla.base, tabla.filas)):
        extra = []
        if con_razones:
            razon = tabla.razones[i]
            extra = ["-" if razon is None else fraccion(razon) + (" ←" if variable == tabla.saliente else "")]
        filas.append([variable, *map(fraccion, fila), *extra])
    filas.append([tabla.etiqueta_objetivo, *map(fraccion, tabla.fila_objetivo), *([""] if con_razones else [])])
    columnas = [c + (" ↓" if c == tabla.entrante else "") for c in tabla.columnas]
    imprimir_tabla(["Base", *columnas, "Solución", *(["Razón"] if con_razones else [])], filas)

    if tabla.entrante is None:
        if tabla.fase == 1:
            print(f"Fin de la fase 1: todos los z_j - c_j >= 0, W* = {fraccion(tabla.fila_objetivo[-1])}.")
        else:
            print("Todos los z_j - c_j son >= 0: la tabla es óptima.")
        return
    k = tabla.columnas.index(tabla.entrante)
    texto = f"Entra {tabla.entrante} (z_j - c_j = {fraccion(tabla.fila_objetivo[k])})"
    if tabla.saliente is None:
        print(f"{texto}, pero ningún coeficiente de su columna es positivo: no hay razón mínima.")
        return
    r = tabla.base.index(tabla.saliente)
    print(f"{texto} · sale {tabla.saliente} (razón mínima {fraccion(tabla.razones[r])}) · "
          f"pivote {fraccion(tabla.filas[r][k])}")


def _imprimir_resultado_lineal(problema: ProblemaLineal, resultado: ResultadoSimplex) -> None:
    nombres = problema.nombres
    if resultado.estado == "infactible":
        w = resultado.tablas[-1].fila_objetivo[-1]
        print(f"\nPROBLEMA INFACTIBLE: la fase 1 terminó con W* = {fraccion(w)} < 0. Alguna variable artificial no "
              "puede valer 0, así que ningún x >= 0 cumple todas las restricciones a la vez.")
    elif resultado.estado == "no_acotado":
        limite = "+∞" if problema.sentido == "max" else "-∞"
        print(f"\nSOLUCIÓN NO ACOTADA: {resultado.variable_no_acotada} puede crecer sin límite sin violar ninguna "
              f"restricción, así que Z → {limite}.")
    else:
        print("\nSolución óptima:")
        imprimir_tabla(["Variable", "Valor"], [[x, f"{v:g}"] for x, v in zip(nombres, resultado.valores)])
        print(f"Z* = {resultado.valor_objetivo:g}\n")
        filas = []
        for i, (r, holgura, precio) in enumerate(zip(problema.restricciones, resultado.holguras, resultado.precios_sombra), 1):
            actividad = sum(a * v for a, v in zip(r.coeficientes, resultado.valores))
            filas.append([
                f"({i})", f"{actividad:g}", f"{r.sentido} {r.lado_derecho:g}", f"{holgura:g}",
                "activa" if abs(holgura) < 1e-9 else "con holgura", f"{precio:g}",
            ])
        imprimir_tabla(["Restricción", "a_i·x", "Límite", "Holgura / exceso", "Estado", "Precio sombra y_i"], filas)
        print("Precio sombra y_i = ∂Z/∂b_i: cuánto cambia Z por cada unidad adicional del lado derecho "
              "(vale 0 si la restricción tiene holgura).")
        if resultado.restricciones_redundantes:
            print("Restricciones redundantes (combinación de las demás): "
                  + ", ".join(f"({i + 1})" for i in resultado.restricciones_redundantes))
        if resultado.degenerada:
            print("La solución es degenerada (una variable básica vale 0): los precios sombra pueden no ser únicos.")
        _imprimir_soluciones_multiples(nombres, resultado)

    estado, _, valor = resolver_con_scipy(problema)
    if estado != resultado.estado:
        print(f"\nATENCIÓN: SciPy (linprog) informa '{estado}' y el Simplex '{resultado.estado}'.")
    elif estado == "optimo":
        coincide = math.isclose(valor, resultado.valor_objetivo, rel_tol=1e-7, abs_tol=1e-7)
        print(f"\nVerificación con SciPy (linprog, HiGHS): Z* = {valor:g} · "
              + ("coincide con el Simplex." if coincide else "NO coincide con el Simplex."))
    else:
        print(f"\nVerificación con SciPy (linprog, HiGHS): también lo declara {estado.replace('_', ' ')}.")


def _imprimir_soluciones_multiples(nombres: Sequence[str], resultado: ResultadoSimplex) -> None:
    if not resultado.soluciones_multiples:
        return
    if resultado.solucion_alternativa is None:
        print("\nSOLUCIONES MÚLTIPLES: una variable no básica tiene z_j - c_j = 0 y ningún coeficiente positivo en "
              "su columna, así que hay infinitas soluciones óptimas a lo largo de una semirrecta.")
        return
    entra, sale = resultado.pivote_alternativo
    print(f"\nSOLUCIONES MÚLTIPLES: {entra} es no básica y tiene z_j - c_j = 0. Si entra (sale {sale}), Z no cambia "
          "y se llega a otra solución óptima:")
    _imprimir_tabla_simplex(resultado.tabla_alternativa, False, "Tabla alternativa")
    imprimir_tabla(
        ["Variable", "Solución X1", "Solución X2"],
        [[x, f"{a:g}", f"{b:g}"] for x, a, b in zip(nombres, resultado.valores, resultado.solucion_alternativa)],
    )
    print(f"Toda combinación λ·X1 + (1 - λ)·X2 con 0 <= λ <= 1 también es óptima (Z* = {resultado.valor_objetivo:g}).")
