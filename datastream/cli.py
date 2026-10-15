"""Punto de entrada: subcomandos (argparse) y menú interactivo con validación."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from . import consola
from .configuracion import Configuracion, ErrorConfiguracion, cargar_configuracion, cargar_variables_entorno

VELOCIDADES = [0.5, 1, 2, 4, 8, 16, 32]


# ====================================================================== #
# Validadores para argparse
# ====================================================================== #
def _numero_en_rango(tipo: type, minimo: float, maximo: float) -> Callable[[str], float]:
    def validar(texto: str):
        try:
            valor = tipo(texto)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{texto}' no es un número {'entero ' if tipo is int else ''}válido")
        if not minimo <= valor <= maximo:
            raise argparse.ArgumentTypeError(f"{valor} está fuera del rango permitido [{minimo}, {maximo}]")
        return valor

    return validar


def _lista_numeros(minimo: float, maximo: float) -> Callable[[str], list[float]]:
    def validar(texto: str) -> list[float]:
        try:
            valores = [float(v) for v in texto.split(",") if v.strip()]
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{texto}' debe ser una lista de números separados por comas")
        if not valores:
            raise argparse.ArgumentTypeError("la lista no puede estar vacía")
        fuera = [v for v in valores if not minimo <= v <= maximo]
        if fuera:
            raise argparse.ArgumentTypeError(f"valores fuera de [{minimo}, {maximo}]: {fuera}")
        return valores

    return validar


# ====================================================================== #
# Acciones
# ====================================================================== #
def ejecutar_simulacion(config: Configuracion, sin_gui: bool, velocidad: float) -> int:
    from .simulacion.centro_datos import SimuladorCentroDatos

    simulador = SimuladorCentroDatos(config)
    if not sin_gui:
        from .visualizacion.dashboard import Dashboard

        Dashboard(simulador, velocidad).ejecutar()
        return 0

    from .reportes.servicio import generar_reporte
    from .simulacion.resumen import construir_resumen

    print(f"Simulando {config.simulacion.duracion:.0f} s sin interfaz gráfica...")
    simulador.ejecutar(lambda avance: print(f"\r  progreso {avance:4.0%}", end="", flush=True))
    m = simulador.metricas
    print(f"\n  generadas {m.generadas} · procesadas {m.procesadas} · descartadas {m.descartadas} · "
          f"espera media {m.espera_promedio:.2f} s · cambios de ruta {m.cambios_ruta}")
    print("Generando reporte PDF (análisis con IA si hay ANTHROPIC_API_KEY)...")
    resultado = generar_reporte(construir_resumen(simulador))
    print(f"Reporte guardado en: {resultado.ruta}  (análisis: {resultado.fuente_analisis})")
    return 0


def _validar_nodos(config: Configuracion, nodos: list[str]) -> None:
    gateway = config.etapas[0][0]
    validos = {n for etapa in config.etapas for n in etapa} - {gateway}
    for nodo in nodos:
        if nodo == gateway:
            raise ErrorConfiguracion(f"El gateway {gateway} no puede fallar (es el único punto de entrada).")
        if nodo not in validos:
            raise ErrorConfiguracion(f"El nodo '{nodo}' no existe. Nodos válidos: {', '.join(sorted(validos))}")


def _validar_cantidad(config: Configuracion, valores: list[float], nombre: str) -> None:
    if len(valores) != len(config.clusters):
        raise ErrorConfiguracion(f"Se esperaban {len(config.clusters)} {nombre} (uno por clúster) y se recibieron {len(valores)}.")


# ====================================================================== #
# Línea de comandos
# ====================================================================== #
def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Simulador de optimización para la red del centro de datos de DataStream Networks. "
        "Sin subcomando se abre un menú interactivo.",
    )
    parser.add_argument("--config", type=Path, help="archivo JSON de configuración (por defecto datos/configuracion_default.json)")
    sub = parser.add_subparsers(dest="comando", metavar="comando")

    p = sub.add_parser("simular", help="ejecuta la simulación (dashboard Pygame o modo sin interfaz + PDF)")
    p.add_argument("--duracion", type=_numero_en_rango(float, 10, 86_400), help="segundos simulados")
    p.add_argument("--semilla", type=_numero_en_rango(int, 0, 2**31 - 1), help="semilla aleatoria")
    p.add_argument("--tasa", type=_numero_en_rango(float, 0.5, 20), help="tasa de llegada base (tareas/s)")
    p.add_argument("--velocidad", type=float, choices=VELOCIDADES, default=4, help="velocidad inicial del dashboard")
    p.add_argument("--sin-gui", action="store_true", help="ejecuta sin ventana y genera el reporte PDF")

    p = sub.add_parser("ruta", help="módulo 1: tabla de programación dinámica de la ruta más corta")
    p.add_argument("--falla", action="append", default=[], metavar="NODO", help="nodo fuera de servicio (repetible)")

    p = sub.add_parser("mochila", help="módulo 1: modelo de volumen-carga")
    p.add_argument("--capacidad", type=_numero_en_rango(int, 1, 200), default=16)
    p.add_argument("--tareas", type=_numero_en_rango(int, 1, 60), default=8)
    p.add_argument("--semilla", type=_numero_en_rango(int, 0, 2**31 - 1), default=7)

    p = sub.add_parser("temperatura", help="módulo 2: puntos estacionarios del modelo térmico")
    p.add_argument("--emergencia", type=_numero_en_rango(float, 0, 200), help="temperatura de emergencia (°C)")

    p = sub.add_parser("gradiente", help="módulo 2: método de gradiente para la refrigeración")
    p.add_argument("--cargas", type=_lista_numeros(0, 1), help="cargas por clúster entre 0 y 1, ej. 0.9,0.5,0.2,0.7")

    p = sub.add_parser("lagrange", help="módulo 2: reparto del presupuesto eléctrico")
    p.add_argument("--presupuesto", type=_numero_en_rango(float, 1, 100_000), help="energía total disponible (kW)")
    p.add_argument("--coeficientes", type=_lista_numeros(0, 100), help="coeficientes a_i por clúster")
    return parser


def ejecutar_comando(config: Configuracion, args: argparse.Namespace) -> int:
    if args.comando == "simular":
        config = config.con_cambios(duracion=args.duracion, semilla=args.semilla, tasa_llegada=args.tasa)
        return ejecutar_simulacion(config, args.sin_gui, args.velocidad)
    if args.comando == "ruta":
        _validar_nodos(config, args.falla)
        consola.demo_ruta(config, args.falla)
    elif args.comando == "mochila":
        consola.demo_mochila(config, args.capacidad, args.tareas, args.semilla)
    elif args.comando == "temperatura":
        consola.demo_temperatura(config, args.emergencia)
    elif args.comando == "gradiente":
        n = len(config.clusters)
        cargas = args.cargas or ([0.9, 0.5, 0.2, 0.7] + [0.5] * n)[:n]
        _validar_cantidad(config, cargas, "cargas")
        consola.demo_gradiente(config, cargas)
    elif args.comando == "lagrange":
        coeficientes = args.coeficientes or consola.coeficientes_por_defecto(config)
        _validar_cantidad(config, coeficientes, "coeficientes")
        if sum(coeficientes) == 0:
            raise ErrorConfiguracion("Al menos un coeficiente a_i debe ser mayor que cero.")
        consola.demo_lagrange(config, args.presupuesto or config.energia.presupuesto_total, coeficientes)
    return 0


# ====================================================================== #
# Menú interactivo
# ====================================================================== #
def pedir_numero(mensaje: str, minimo: float, maximo: float, defecto: float, entero: bool = False) -> float:
    """Pide un número hasta que sea válido. Enter acepta el valor por defecto."""
    while True:
        texto = input(f"{mensaje} [{minimo}-{maximo}] (Enter = {defecto}): ").strip()
        if not texto:
            return defecto
        try:
            valor = int(texto) if entero else float(texto.replace(",", "."))
        except ValueError:
            print(f"  ✗ '{texto}' no es un número {'entero ' if entero else ''}válido.")
            continue
        if minimo <= valor <= maximo:
            return valor
        print(f"  ✗ El valor debe estar entre {minimo} y {maximo}.")


def pedir_lista(mensaje: str, cantidad: int, minimo: float, maximo: float, defecto: list[float]) -> list[float]:
    texto_defecto = ",".join(f"{v:g}" for v in defecto)
    while True:
        texto = input(f"{mensaje} ({cantidad} valores separados por coma, Enter = {texto_defecto}): ").strip()
        if not texto:
            return defecto
        try:
            valores = [float(v) for v in texto.split(",")]
        except ValueError:
            print("  ✗ Use solo números separados por comas (ej. 0.8,0.5,0.3,0.9).")
            continue
        if len(valores) != cantidad:
            print(f"  ✗ Debe ingresar exactamente {cantidad} valores.")
        elif any(not minimo <= v <= maximo for v in valores):
            print(f"  ✗ Todos los valores deben estar entre {minimo} y {maximo}.")
        else:
            return valores


def pedir_si_no(mensaje: str, defecto: bool = True) -> bool:
    while True:
        texto = input(f"{mensaje} (s/n, Enter = {'s' if defecto else 'n'}): ").strip().lower()
        if not texto:
            return defecto
        if texto in ("s", "si", "sí"):
            return True
        if texto in ("n", "no"):
            return False
        print("  ✗ Responda 's' o 'n'.")


OPCIONES = """
 1. Ejecutar simulación con dashboard (Pygame)
 2. Ejecutar simulación sin interfaz y generar reporte PDF
 3. Módulo 1 · Ruta más corta (programación dinámica)
 4. Módulo 1 · Modelo de volumen-carga (mochila)
 5. Módulo 2 · Análisis térmico (puntos estacionarios)
 6. Módulo 2 · Método de gradiente (refrigeración)
 7. Módulo 2 · Lagrange (presupuesto eléctrico)
 8. Cargar otro archivo de configuración
 0. Salir"""


def menu_interactivo(config: Configuracion, ruta_config: Path | None) -> int:
    while True:
        consola.titulo(f"DataStream Networks · Simulador  (configuración: {ruta_config or 'datos por defecto'})")
        print(OPCIONES)
        opcion = input("\nSeleccione una opción: ").strip()
        try:
            if opcion == "0":
                return 0
            if opcion in ("1", "2"):
                duracion = pedir_numero("Duración simulada (s)", 10, 86_400, config.simulacion.duracion)
                semilla = pedir_numero("Semilla aleatoria", 0, 2**31 - 1, config.simulacion.semilla, entero=True)
                tasa = pedir_numero("Tasa de llegada base (tareas/s)", 0.5, 20, config.simulacion.tasa_llegada)
                ajustada = config.con_cambios(duracion=duracion, semilla=semilla, tasa_llegada=tasa)
                ejecutar_simulacion(ajustada, sin_gui=opcion == "2", velocidad=4)
            elif opcion == "3":
                texto = input("Nodos caídos separados por coma (Enter = ninguno): ").strip()
                nodos = [n.strip() for n in texto.split(",") if n.strip()]
                _validar_nodos(config, nodos)
                consola.demo_ruta(config, nodos)
            elif opcion == "4":
                capacidad = pedir_numero("Capacidad del servidor (CPU)", 1, 200, 16, entero=True)
                tareas = pedir_numero("Cantidad de tareas en cola", 1, 60, 8, entero=True)
                semilla = pedir_numero("Semilla", 0, 10_000, 7, entero=True)
                consola.demo_mochila(config, capacidad, tareas, semilla)
            elif opcion == "5":
                umbral = pedir_numero("Temperatura de emergencia (°C)", 0, 200, config.termico.temperatura_emergencia)
                consola.demo_temperatura(config, umbral)
            elif opcion == "6":
                n = len(config.clusters)
                cargas = pedir_lista("Cargas de los clústeres entre 0 y 1", n, 0, 1, ([0.9, 0.5, 0.2, 0.7] + [0.5] * n)[:n])
                consola.demo_gradiente(config, cargas)
            elif opcion == "7":
                presupuesto = pedir_numero("Presupuesto eléctrico (kW)", 1, 100_000, config.energia.presupuesto_total)
                defecto = [round(a, 3) for a in consola.coeficientes_por_defecto(config)]
                coeficientes = pedir_lista("Coeficientes a_i", len(config.clusters), 0, 100, defecto)
                if sum(coeficientes) == 0:
                    raise ErrorConfiguracion("Al menos un coeficiente a_i debe ser mayor que cero.")
                consola.demo_lagrange(config, presupuesto, coeficientes)
            elif opcion == "8":
                ruta = Path(input("Ruta del archivo JSON: ").strip().strip('"'))
                config, ruta_config = cargar_configuracion(ruta), ruta
                print("  ✓ Configuración cargada y validada correctamente.")
            else:
                print("  ✗ Opción inválida: ingrese un número del 0 al 8.")
                continue
        except ErrorConfiguracion as error:
            print(f"  ✗ {error}")
        input("\nPresione Enter para volver al menú...")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # consolas sin UTF-8 (Windows antiguo)
    cargar_variables_entorno()
    args = construir_parser().parse_args(argv)
    try:
        config = cargar_configuracion(args.config)
        if args.comando is None:
            return menu_interactivo(config, args.config)
        return ejecutar_comando(config, args)
    except ErrorConfiguracion as error:
        print(f"Error de datos: {error}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("\nEjecución cancelada por el usuario.")
        return 130
