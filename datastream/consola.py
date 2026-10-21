"""Ejecución independiente de cada módulo con salida por consola."""

from __future__ import annotations

import math
import random
from typing import Sequence

from .configuracion import Configuracion
from .modelos.mochila import Articulo, ModeloMochilaUnidades, ModeloVolumenCarga, ResultadoMochilaUnidades
from .modelos.presupuesto import AsignadorEnergia
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


def _imprimir_hacia_adelante(resultado: ResultadoRutas) -> None:
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
    imprimir_tabla(["Etapa", "Nodo y", "Candidatos x: f(x)+d(x,y)", "f_k(y)", "Decisión x*"], filas)


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


def demo_ruta(config: Configuracion, nodos_caidos: Sequence[str] = (), destino: str | None = None) -> None:
    titulo("MÓDULO 1 · Ruta más corta con programación dinámica")
    etapas = [list(e) for e in config.etapas]
    caidos = set(nodos_caidos)
    arcos = {
        (e.origen, e.destino): e.latencia
        for e in config.enlaces
        if e.origen not in caidos and e.destino not in caidos
    }
    if caidos:
        print(f"Nodos fuera de servicio: {', '.join(sorted(caidos))}")
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


def demo_lagrange(config: Configuracion, presupuesto: float, coeficientes: Sequence[float]) -> None:
    titulo("MÓDULO 2 · Asignación del presupuesto eléctrico con multiplicadores de Lagrange")
    asignador = AsignadorEnergia(presupuesto)
    simbolico = asignador.derivacion_simbolica(len(coeficientes))
    print(f"L(E, λ) = {simbolico['lagrangiano']}\n")
    print("Condiciones de primer orden (derivadas parciales = 0):")
    for condicion in simbolico["condiciones"]:
        print(f"   {condicion.lhs} = 0")
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
    print("Como ∂²R/∂E_i² = -a_i/(4·E_i^1.5) < 0, la función es cóncava y el punto es un máximo.")


def coeficientes_por_defecto(config: Configuracion, carga: float = 0.5) -> list[float]:
    """a_i que usaría la simulación con todos los servidores activos y carga uniforme."""
    capacidad_total = sum(s.capacidad for c in config.clusters for s in c.servidores)
    return [
        c.eficiencia * (sum(s.capacidad for s in c.servidores) / capacidad_total) * (config.energia.carga_base + carga)
        for c in config.clusters
    ]
