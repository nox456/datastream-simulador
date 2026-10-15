"""Carga y validación de la configuración del simulador (archivo JSON).

Todos los datos que introduce el usuario pasan por aquí: si algo no es válido
se lanza ErrorConfiguracion con un mensaje que indica el campo y el motivo.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .dominio.tarea import TipoTarea
from .modelos.temperatura import ModeloTermico

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIG_DEFECTO = RAIZ_PROYECTO / "datos" / "configuracion_default.json"


class ErrorConfiguracion(ValueError):
    """Dato de configuración inválido."""


@dataclass(frozen=True)
class ParametrosSimulacion:
    duracion: float
    semilla: int
    tasa_llegada: float
    amplitud_pico: float
    periodo_pico: float
    intervalo_despacho: float
    intervalo_rutas: float
    intervalo_control: float
    espera_maxima: float
    factor_congestion: float
    suavizado_congestion: float


@dataclass(frozen=True)
class ConfigEnlace:
    origen: str
    destino: str
    latencia: float


@dataclass(frozen=True)
class ConfigServidor:
    id: str
    capacidad: int


@dataclass(frozen=True)
class ConfigCluster:
    nombre: str
    switch: str
    eficiencia: float
    latencia_acceso: float
    servidores: tuple[ConfigServidor, ...]


@dataclass(frozen=True)
class ParametrosFallas:
    mtbf_router: float
    mttr_router: float
    mtbf_servidor: float
    mttr_servidor: float


@dataclass(frozen=True)
class ParametrosTermicos:
    funcion: str
    carga_min: float
    carga_max: float
    temperatura_emergencia: float
    inercia: float  # constante de tiempo térmica τ (s)


@dataclass(frozen=True)
class ParametrosRefrigeracion:
    coef_ventilador: float
    coef_calor: float
    coef_acople: float
    tasa_aprendizaje: float
    tolerancia: float
    max_iteraciones: int
    nivel_minimo: float


@dataclass(frozen=True)
class ParametrosEnergia:
    presupuesto_total: float
    carga_base: float


@dataclass(frozen=True)
class Configuracion:
    simulacion: ParametrosSimulacion
    etapas: tuple[tuple[str, ...], ...]
    enlaces: tuple[ConfigEnlace, ...]
    clusters: tuple[ConfigCluster, ...]
    tipos_tarea: tuple[TipoTarea, ...]
    fallas: ParametrosFallas
    termico: ParametrosTermicos
    refrigeracion: ParametrosRefrigeracion
    energia: ParametrosEnergia

    def con_cambios(self, **cambios_simulacion: Any) -> Configuracion:
        """Copia con parámetros de simulación sobrescritos (p. ej. desde la CLI)."""
        validos = {k: v for k, v in cambios_simulacion.items() if v is not None}
        return replace(self, simulacion=replace(self.simulacion, **validos))


# ---------------------------------------------------------------------- #
# Funciones auxiliares de validación
# ---------------------------------------------------------------------- #
def _seccion(datos: dict, clave: str) -> dict:
    valor = datos.get(clave)
    if not isinstance(valor, dict):
        raise ErrorConfiguracion(f"Falta la sección '{clave}' o no es un objeto JSON.")
    return valor


def _numero(
    datos: dict,
    clave: str,
    ruta: str,
    minimo: float | None = None,
    maximo: float | None = None,
    entero: bool = False,
    excluir_minimo: bool = False,
) -> float:
    valor = datos.get(clave)
    nombre = f"{ruta}.{clave}"
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
        raise ErrorConfiguracion(f"'{nombre}' debe ser un número (recibido: {valor!r}).")
    if entero and int(valor) != valor:
        raise ErrorConfiguracion(f"'{nombre}' debe ser un número entero (recibido: {valor}).")
    if minimo is not None and (valor < minimo or (excluir_minimo and valor == minimo)):
        signo = ">" if excluir_minimo else ">="
        raise ErrorConfiguracion(f"'{nombre}' debe ser {signo} {minimo} (recibido: {valor}).")
    if maximo is not None and valor > maximo:
        raise ErrorConfiguracion(f"'{nombre}' debe ser <= {maximo} (recibido: {valor}).")
    return int(valor) if entero else float(valor)


def _positivo(datos: dict, clave: str, ruta: str, maximo: float | None = None) -> float:
    return _numero(datos, clave, ruta, minimo=0, maximo=maximo, excluir_minimo=True)


def _texto(datos: dict, clave: str, ruta: str) -> str:
    valor = datos.get(clave)
    if not isinstance(valor, str) or not valor.strip():
        raise ErrorConfiguracion(f"'{ruta}.{clave}' debe ser un texto no vacío.")
    return valor.strip()


def _rango_enteros(datos: dict, clave: str, ruta: str, minimo: int) -> tuple[int, int]:
    valor = datos.get(clave)
    nombre = f"{ruta}.{clave}"
    if (
        not isinstance(valor, list)
        or len(valor) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in valor)
    ):
        raise ErrorConfiguracion(f"'{nombre}' debe ser una lista [mínimo, máximo] de enteros.")
    bajo, alto = valor
    if bajo < minimo or bajo > alto:
        raise ErrorConfiguracion(f"'{nombre}' debe cumplir {minimo} <= mínimo <= máximo.")
    return bajo, alto


# ---------------------------------------------------------------------- #
# Lectura de cada sección
# ---------------------------------------------------------------------- #
def _leer_simulacion(datos: dict) -> ParametrosSimulacion:
    s, r = _seccion(datos, "simulacion"), "simulacion"
    return ParametrosSimulacion(
        duracion=_positivo(s, "duracion", r, maximo=86_400),
        semilla=_numero(s, "semilla", r, minimo=0, entero=True),
        tasa_llegada=_positivo(s, "tasa_llegada", r, maximo=50),
        amplitud_pico=_numero(s, "amplitud_pico", r, minimo=0, maximo=0.95),
        periodo_pico=_positivo(s, "periodo_pico", r),
        intervalo_despacho=_positivo(s, "intervalo_despacho", r, maximo=10),
        intervalo_rutas=_positivo(s, "intervalo_rutas", r, maximo=10),
        intervalo_control=_positivo(s, "intervalo_control", r, maximo=10),
        espera_maxima=_positivo(s, "espera_maxima", r),
        factor_congestion=_numero(s, "factor_congestion", r, minimo=0, maximo=5),
        suavizado_congestion=_numero(s, "suavizado_congestion", r, minimo=0, maximo=1, excluir_minimo=True),
    )


def _leer_topologia(datos: dict) -> tuple[tuple[tuple[str, ...], ...], tuple[ConfigEnlace, ...]]:
    t = _seccion(datos, "topologia")
    etapas = t.get("etapas")
    if not isinstance(etapas, list) or len(etapas) < 2:
        raise ErrorConfiguracion("'topologia.etapas' debe ser una lista con al menos 2 etapas.")
    vistos: set[str] = set()
    etapa_de: dict[str, int] = {}
    for k, etapa in enumerate(etapas):
        if not isinstance(etapa, list) or not etapa or not all(isinstance(n, str) and n for n in etapa):
            raise ErrorConfiguracion(f"La etapa {k} debe ser una lista no vacía de identificadores.")
        for nodo in etapa:
            if nodo in vistos:
                raise ErrorConfiguracion(f"El nodo '{nodo}' está repetido en la topología.")
            vistos.add(nodo)
            etapa_de[nodo] = k
    if len(etapas[0]) != 1:
        raise ErrorConfiguracion("La etapa 0 debe tener un único nodo (el gateway de entrada).")

    lista = t.get("enlaces")
    if not isinstance(lista, list) or not lista:
        raise ErrorConfiguracion("'topologia.enlaces' debe ser una lista no vacía.")
    enlaces, claves = [], set()
    for i, e in enumerate(lista):
        ruta = f"topologia.enlaces[{i}]"
        if not isinstance(e, dict):
            raise ErrorConfiguracion(f"'{ruta}' debe ser un objeto JSON.")
        origen, destino = _texto(e, "origen", ruta), _texto(e, "destino", ruta)
        for nodo in (origen, destino):
            if nodo not in etapa_de:
                raise ErrorConfiguracion(f"'{ruta}' usa el nodo desconocido '{nodo}'.")
        if etapa_de[destino] != etapa_de[origen] + 1:
            raise ErrorConfiguracion(f"'{ruta}': {origen}→{destino} debe unir etapas consecutivas.")
        if (origen, destino) in claves:
            raise ErrorConfiguracion(f"'{ruta}': el enlace {origen}→{destino} está repetido.")
        claves.add((origen, destino))
        enlaces.append(ConfigEnlace(origen, destino, _positivo(e, "latencia", ruta, maximo=60)))

    # Cada nodo (salvo el gateway) debe tener al menos un enlace de entrada.
    con_entrada = {destino for _, destino in claves}
    for nodo, k in etapa_de.items():
        if k > 0 and nodo not in con_entrada:
            raise ErrorConfiguracion(f"El nodo '{nodo}' no tiene enlaces de entrada (inalcanzable).")
    return tuple(tuple(e) for e in etapas), tuple(enlaces)


def _leer_clusters(datos: dict, ultima_etapa: tuple[str, ...], nodos_red: set[str]) -> tuple[ConfigCluster, ...]:
    lista = datos.get("clusters")
    if not isinstance(lista, list) or not lista:
        raise ErrorConfiguracion("'clusters' debe ser una lista no vacía.")
    clusters, nombres, switches, ids_servidor = [], set(), set(), set()
    for i, c in enumerate(lista):
        ruta = f"clusters[{i}]"
        if not isinstance(c, dict):
            raise ErrorConfiguracion(f"'{ruta}' debe ser un objeto JSON.")
        nombre, switch = _texto(c, "nombre", ruta), _texto(c, "switch", ruta)
        if nombre in nombres:
            raise ErrorConfiguracion(f"El nombre de clúster '{nombre}' está repetido.")
        if switch not in ultima_etapa:
            raise ErrorConfiguracion(f"'{ruta}.switch' ({switch}) debe ser un nodo de la última etapa de la red.")
        if switch in switches:
            raise ErrorConfiguracion(f"El switch '{switch}' ya está asignado a otro clúster.")
        nombres.add(nombre)
        switches.add(switch)

        servidores = c.get("servidores")
        if not isinstance(servidores, list) or not servidores:
            raise ErrorConfiguracion(f"'{ruta}.servidores' debe ser una lista no vacía.")
        lista_servidores = []
        for j, s in enumerate(servidores):
            ruta_s = f"{ruta}.servidores[{j}]"
            if not isinstance(s, dict):
                raise ErrorConfiguracion(f"'{ruta_s}' debe ser un objeto JSON.")
            identificador = _texto(s, "id", ruta_s)
            if identificador in ids_servidor or identificador in nodos_red:
                raise ErrorConfiguracion(f"El identificador '{identificador}' ya está en uso.")
            ids_servidor.add(identificador)
            capacidad = _numero(s, "capacidad", ruta_s, minimo=1, maximo=1024, entero=True)
            lista_servidores.append(ConfigServidor(identificador, capacidad))

        clusters.append(
            ConfigCluster(
                nombre=nombre,
                switch=switch,
                eficiencia=_positivo(c, "eficiencia", ruta, maximo=10),
                latencia_acceso=_positivo(c, "latencia_acceso", ruta, maximo=60),
                servidores=tuple(lista_servidores),
            )
        )
    if len(clusters) > 6:
        raise ErrorConfiguracion("Se admiten como máximo 6 clústeres (límite del dashboard).")
    return tuple(clusters)


def _leer_tipos_tarea(datos: dict, capacidad_minima: int) -> tuple[TipoTarea, ...]:
    lista = datos.get("tipos_tarea")
    if not isinstance(lista, list) or not lista:
        raise ErrorConfiguracion("'tipos_tarea' debe ser una lista no vacía.")
    tipos = []
    for i, t in enumerate(lista):
        ruta = f"tipos_tarea[{i}]"
        if not isinstance(t, dict):
            raise ErrorConfiguracion(f"'{ruta}' debe ser un objeto JSON.")
        cpu_min, cpu_max = _rango_enteros(t, "cpu", ruta, minimo=1)
        if cpu_max > capacidad_minima:
            raise ErrorConfiguracion(
                f"'{ruta}.cpu' máximo ({cpu_max}) supera la capacidad del servidor más pequeño ({capacidad_minima})."
            )
        prioridad_min, prioridad_max = _rango_enteros(t, "prioridad", ruta, minimo=1)
        if prioridad_max > 10:
            raise ErrorConfiguracion(f"'{ruta}.prioridad' debe estar entre 1 y 10.")
        tipos.append(
            TipoTarea(
                nombre=_texto(t, "nombre", ruta),
                probabilidad=_positivo(t, "probabilidad", ruta, maximo=1),
                cpu_min=cpu_min,
                cpu_max=cpu_max,
                prioridad_min=prioridad_min,
                prioridad_max=prioridad_max,
                tiempo_servicio=_positivo(t, "tiempo_servicio", ruta, maximo=3600),
            )
        )
    total = sum(t.probabilidad for t in tipos)
    if abs(total - 1.0) > 0.01:
        raise ErrorConfiguracion(f"Las probabilidades de 'tipos_tarea' deben sumar 1 (suman {total:.3f}).")
    return tuple(tipos)


def _leer_termico(datos: dict) -> ParametrosTermicos:
    t, r = _seccion(datos, "temperatura"), "temperatura"
    parametros = ParametrosTermicos(
        funcion=_texto(t, "funcion", r),
        carga_min=_numero(t, "carga_min", r, minimo=0, maximo=100),
        carga_max=_numero(t, "carga_max", r, minimo=0, maximo=100),
        temperatura_emergencia=_numero(t, "temperatura_emergencia", r, minimo=0, maximo=200),
        inercia=_positivo(t, "inercia", r, maximo=3600),
    )
    if parametros.carga_min >= parametros.carga_max:
        raise ErrorConfiguracion("'temperatura.carga_min' debe ser menor que 'carga_max'.")
    try:
        ModeloTermico(parametros.funcion, (parametros.carga_min, parametros.carga_max))
    except ValueError as error:
        raise ErrorConfiguracion(f"'temperatura.funcion': {error}") from error
    return parametros


def _leer_refrigeracion(datos: dict) -> ParametrosRefrigeracion:
    s, r = _seccion(datos, "refrigeracion"), "refrigeracion"
    return ParametrosRefrigeracion(
        coef_ventilador=_positivo(s, "coef_ventilador", r),
        coef_calor=_positivo(s, "coef_calor", r),
        coef_acople=_numero(s, "coef_acople", r, minimo=0),
        tasa_aprendizaje=_positivo(s, "tasa_aprendizaje", r, maximo=1),
        tolerancia=_positivo(s, "tolerancia", r, maximo=1),
        max_iteraciones=_numero(s, "max_iteraciones", r, minimo=1, maximo=10_000, entero=True),
        nivel_minimo=_numero(s, "nivel_minimo", r, minimo=0, maximo=0.9, excluir_minimo=True),
    )


def _leer_energia(datos: dict) -> ParametrosEnergia:
    s, r = _seccion(datos, "energia"), "energia"
    return ParametrosEnergia(
        presupuesto_total=_positivo(s, "presupuesto_total", r, maximo=100_000),
        carga_base=_numero(s, "carga_base", r, minimo=0, maximo=1),
    )


def _leer_fallas(datos: dict) -> ParametrosFallas:
    s, r = _seccion(datos, "fallas"), "fallas"
    return ParametrosFallas(
        mtbf_router=_positivo(s, "mtbf_router", r),
        mttr_router=_positivo(s, "mttr_router", r),
        mtbf_servidor=_positivo(s, "mtbf_servidor", r),
        mttr_servidor=_positivo(s, "mttr_servidor", r),
    )


# ---------------------------------------------------------------------- #
# API pública
# ---------------------------------------------------------------------- #
def construir_configuracion(datos: Any) -> Configuracion:
    if not isinstance(datos, dict):
        raise ErrorConfiguracion("El archivo de configuración debe contener un objeto JSON.")
    etapas, enlaces = _leer_topologia(datos)
    nodos_red = {nodo for etapa in etapas for nodo in etapa}
    clusters = _leer_clusters(datos, etapas[-1], nodos_red)
    capacidad_minima = min(s.capacidad for c in clusters for s in c.servidores)
    return Configuracion(
        simulacion=_leer_simulacion(datos),
        etapas=etapas,
        enlaces=enlaces,
        clusters=clusters,
        tipos_tarea=_leer_tipos_tarea(datos, capacidad_minima),
        fallas=_leer_fallas(datos),
        termico=_leer_termico(datos),
        refrigeracion=_leer_refrigeracion(datos),
        energia=_leer_energia(datos),
    )


def cargar_configuracion(ruta: str | Path | None = None) -> Configuracion:
    """Lee y valida un archivo JSON. Sin ruta se usan los datos por defecto."""
    ruta = Path(ruta) if ruta else RUTA_CONFIG_DEFECTO
    if not ruta.is_file():
        raise ErrorConfiguracion(f"No existe el archivo de configuración: {ruta}")
    if ruta.suffix.lower() != ".json":
        raise ErrorConfiguracion("El archivo de configuración debe tener extensión .json")
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ErrorConfiguracion(
            f"JSON inválido en {ruta.name} (línea {error.lineno}, columna {error.colno}): {error.msg}"
        ) from error
    return construir_configuracion(datos)


def cargar_variables_entorno(ruta: Path = RAIZ_PROYECTO / ".env") -> None:
    """Carga variables KEY=VALOR desde un archivo .env (sin pisar las existentes)."""
    if not ruta.is_file():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))
