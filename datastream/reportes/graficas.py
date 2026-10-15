"""Gráficas estáticas (PNG) para el reporte PDF, generadas con matplotlib.

Se usa la API orientada a objetos (Figure + FigureCanvasAgg) en lugar de
pyplot porque el reporte se genera en un hilo aparte del dashboard.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..modelos.temperatura import ModeloTermico

COLORES = ["#1f6feb", "#2da44e", "#8250df", "#d1242f", "#bf8700", "#1b7c83"]


def _figura(filas: int = 1, alto: float = 3.2) -> tuple[Figure, list]:
    figura = Figure(figsize=(7.2, alto * filas), layout="constrained")
    FigureCanvasAgg(figura)
    ejes = [figura.add_subplot(filas, 1, i + 1) for i in range(filas)]
    for eje in ejes:
        eje.grid(alpha=0.3)
    return figura, ejes


def _guardar(figura: Figure, ruta: Path) -> Path:
    figura.savefig(ruta, dpi=140)
    return ruta


def grafica_trafico(resumen: dict, ruta: Path) -> Path:
    s = resumen["series"]
    figura, (arriba, abajo) = _figura(2, 2.6)
    arriba.plot(s["t"], s["tasa_llegada"], color=COLORES[3], label="λ(t) llegadas")
    arriba.plot(s["t"], s["throughput"], color=COLORES[0], alpha=0.6, label="throughput")
    arriba.set_ylabel("tareas/s")
    arriba.set_title("Llegadas (Poisson con hora pico) y tareas completadas")
    arriba.legend(loc="upper right", fontsize=8)
    abajo.plot(s["t"], s["en_cola"], color=COLORES[4], label="tareas en cola")
    eje_util = abajo.twinx()
    eje_util.plot(s["t"], [u * 100 for u in s["utilizacion"]], color=COLORES[1], alpha=0.7, label="utilización %")
    eje_util.set_ylabel("utilización (%)")
    abajo.set_ylabel("tareas en cola")
    abajo.set_xlabel("tiempo simulado (s)")
    abajo.set_title("Cola de despacho y utilización de la capacidad")
    return _guardar(figura, ruta)


def grafica_refrigeracion(resumen: dict, ruta: Path) -> Path:
    s, nombres = resumen["series"], resumen["parametros"]["clusters"]
    figura, (arriba, abajo) = _figura(2, 2.6)
    for i, nombre in enumerate(nombres):
        arriba.plot(s["t"], [v[i] * 100 for v in s["niveles_ventilador"]], color=COLORES[i % 6], label=nombre)
    arriba.set_ylabel("potencia ventiladores (%)")
    arriba.set_title("Método de gradiente: ajuste continuo de la ventilación")
    arriba.legend(loc="upper right", fontsize=7, ncol=len(nombres))
    eje_e = arriba.twinx()
    eje_e.plot(s["t"], s["energia_refrigeracion"], color="black", alpha=0.35, linewidth=1)
    eje_e.set_ylabel("E(u*) kW", color="gray")
    historial = s["historial_gradiente"]
    abajo.plot(range(len(historial)), historial, marker="o", markersize=3, color=COLORES[4])
    abajo.set_xlabel("iteración k")
    abajo.set_ylabel("E(u^k) kW")
    abajo.set_title("Convergencia del último ciclo de optimización")
    return _guardar(figura, ruta)


def grafica_lagrange(resumen: dict, ruta: Path) -> Path:
    s, nombres = resumen["series"], resumen["parametros"]["clusters"]
    figura, (arriba, abajo) = _figura(2, 2.6)
    capas = [[v[i] for v in s["energias_cluster"]] for i in range(len(nombres))]
    arriba.stackplot(s["t"], capas, labels=nombres, colors=COLORES[: len(nombres)], alpha=0.8)
    arriba.set_ylabel("energía asignada (kW)")
    arriba.set_title("Reparto del presupuesto eléctrico (Σ E_i = E_max)")
    arriba.legend(loc="upper right", fontsize=7, ncol=len(nombres))
    abajo.plot(s["t"], s["multiplicador"], color=COLORES[3])
    abajo.set_xlabel("tiempo simulado (s)")
    abajo.set_ylabel("λ* (rendimiento / kW)")
    abajo.set_title("Multiplicador de Lagrange (precio sombra)")
    return _guardar(figura, ruta)


def grafica_temperatura(resumen: dict, ruta: Path) -> Path:
    term, s, nombres = resumen["termico"], resumen["series"], resumen["parametros"]["clusters"]
    x_min, x_max = term["dominio"]
    modelo = ModeloTermico(term["funcion"], (x_min, x_max), term["temperatura_emergencia"])
    figura, (arriba, abajo) = _figura(2, 2.8)
    cargas = [x_min + (x_max - x_min) * i / 200 for i in range(201)]
    arriba.plot(cargas, [modelo.temperatura(c) for c in cargas], color="black", label="T(x)")
    marcadores = {"máximo local": ("^", COLORES[3]), "mínimo local": ("v", COLORES[1])}
    for p in term["estacionarios"]:
        forma, color = marcadores.get(p["tipo"], ("D", COLORES[4]))
        arriba.plot(p["carga"], p["temperatura"], forma, color=color, markersize=9, label=f"{p['tipo']} ({p['carga']:.1f}%)")
    for p in term["inflexiones"]:
        arriba.plot(p["carga"], p["temperatura"], "D", color=COLORES[4], markersize=7, label=f"inflexión ({p['carga']:.1f}%)")
    arriba.axhline(term["temperatura_emergencia"], color=COLORES[3], linestyle="--", linewidth=1)
    if term["carga_emergencia"] is not None:
        arriba.axvspan(term["carga_emergencia"], x_max, color=COLORES[3], alpha=0.1, label="zona de emergencia")
    arriba.set_xlabel("carga x (%)")
    arriba.set_ylabel("T (°C)")
    arriba.set_title("Modelo térmico y puntos notables (T'(x)=0, T''(x)=0)")
    arriba.legend(fontsize=7, loc="upper left")
    for i, nombre in enumerate(nombres):
        abajo.plot(s["t"], [v[i] for v in s["temperaturas"]], color=COLORES[i % 6], label=nombre)
    abajo.axhline(term["temperatura_emergencia"], color=COLORES[3], linestyle="--", linewidth=1, label="umbral")
    abajo.set_xlabel("tiempo simulado (s)")
    abajo.set_ylabel("T máx. (°C)")
    abajo.set_title("Temperatura máxima por clúster durante la simulación")
    abajo.legend(fontsize=7, ncol=len(nombres) + 1, loc="upper left")
    return _guardar(figura, ruta)


def generar_graficas(resumen: dict, directorio: Path) -> dict[str, Path]:
    directorio.mkdir(parents=True, exist_ok=True)
    return {
        "trafico": grafica_trafico(resumen, directorio / "trafico.png"),
        "refrigeracion": grafica_refrigeracion(resumen, directorio / "refrigeracion.png"),
        "lagrange": grafica_lagrange(resumen, directorio / "lagrange.png"),
        "temperatura": grafica_temperatura(resumen, directorio / "temperatura.png"),
    }
