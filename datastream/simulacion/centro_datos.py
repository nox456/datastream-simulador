"""Simulación de eventos discretos del centro de datos (SimPy).

Procesos que se ejecutan en paralelo dentro del entorno de SimPy:

    _llegadas        Llegada aleatoria de tareas: proceso de Poisson con tasa
                     variable λ(t) (hora pico), generado por "thinning".
    _despachar       Cada Δt resuelve el modelo de volumen-carga (mochila) para
                     decidir qué tareas de la cola recibe cada servidor.
    _transmitir      Viaje del paquete enlace por enlace por la ruta más corta.
    _procesar        Tiempo de servicio exponencial en el servidor.
    _ciclo_fallas    Fallas y reparaciones estocásticas de cada nodo (MTBF/MTTR).
    rutas (periódico)    Programación dinámica con la congestión del momento.
    control (periódico)  Modelo térmico + método de gradiente + Lagrange.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable

import simpy

from ..configuracion import Configuracion
from ..dominio.red import Enlace, Nodo, TipoNodo, Topologia
from ..dominio.servidor import Cluster, Servidor
from ..dominio.tarea import PaqueteEnTransito, Tarea
from ..modelos.bellman_ford import BellmanFordDP
from ..modelos.mochila import Articulo, ModeloVolumenCarga
from ..modelos.presupuesto import AsignadorEnergia, ResultadoLagrange
from ..modelos.refrigeracion import OptimizadorRefrigeracion, ResultadoGradiente
from ..modelos.ruta_mas_corta import ResultadoRutas, RutaMasCortaDP
from ..modelos.temperatura import ModeloTermico
from .metricas import Metricas, Muestra

TASA_MINIMA, TASA_MAXIMA = 0.5, 20.0  # límites para el ajuste en vivo de λ
UMBRAL_CAMBIO_RUTA = 0.08  # la nueva ruta debe ser 8 % más rápida para reemplazar a la actual
HISTERESIS_TERMICA = 5.0  # °C bajo el umbral para salir de la emergencia (evita oscilar)
MAX_ARTICULOS_ILUSTRATIVOS = 15  # tamaño máximo de la mochila que se muestra en el reporte


@dataclass(frozen=True)
class CambioRuta:
    t: float
    cluster: str
    anterior: tuple[str, ...] | None
    nueva: tuple[str, ...] | None
    motivo: str
    latencia: float


@dataclass(frozen=True)
class DecisionMochila:
    """Instancia resuelta del modelo de volumen-carga (para dashboard y reporte)."""

    t: float
    servidor: str
    capacidad: int
    articulos: tuple[Articulo, ...]
    seleccionados: tuple[int, ...]
    valor_total: float
    peso_total: int


class SimuladorCentroDatos:
    def __init__(self, config: Configuracion):
        self.config = config
        p = config.simulacion
        self.env = simpy.Environment()
        self.azar = random.Random(p.semilla)
        self.tasa_llegada = p.tasa_llegada  # λ base (ajustable durante la ejecución)
        self.metricas = Metricas()
        self._construir_red()

        # --- Modelos matemáticos ------------------------------------------------
        self.enrutador = self._crear_enrutador()
        self.mochila = ModeloVolumenCarga()
        t = config.termico
        self.modelo_termico = ModeloTermico(t.funcion, (t.carga_min, t.carga_max), t.temperatura_emergencia)
        self.analisis_termico = self.modelo_termico.analizar()
        for servidor in self.servidores:
            servidor.temperatura = self.modelo_termico.temperatura(0)  # arranca en reposo
        r = config.refrigeracion
        self.refrigeracion = OptimizadorRefrigeracion(
            r.coef_ventilador,
            r.coef_calor,
            r.coef_acople,
            vecinos=[(i, i + 1) for i in range(len(self.clusters) - 1)],  # pasillos contiguos
            tasa_aprendizaje=r.tasa_aprendizaje,
            tolerancia=r.tolerancia,
            max_iteraciones=r.max_iteraciones,
            nivel_minimo=r.nivel_minimo,
        )
        self.asignador = AsignadorEnergia(config.energia.presupuesto_total)

        # --- Estado dinámico ----------------------------------------------------
        self.cola: list[Tarea] = []
        self.paquetes: list[PaqueteEnTransito] = []
        self.tareas_en_transito = 0
        self.rutas_actuales: dict[str, tuple[str, ...] | None] = {}
        self.latencias_actuales: dict[str, float] = {}
        # La ruta en uso puede diferir de la óptima de la DP por la regla anti-oscilación.
        self.rutas_optimas: dict[str, tuple[str, ...] | None] = {}
        self.latencias_optimas: dict[str, float] = {}
        self.cambios_ruta: list[CambioRuta] = []
        self.resultado_rutas: ResultadoRutas | None = None
        self.ultima_mochila: DecisionMochila | None = None
        self.mochila_destacada: DecisionMochila | None = None  # instancia ilustrativa para el reporte
        self.ultimo_gradiente: ResultadoGradiente | None = None
        self.ultimo_lagrange: ResultadoLagrange | None = None
        self.ultimos_coeficientes: tuple[float, ...] = ()
        self._ids = itertools.count(1)
        self._procesadas_previas = 0

        # Estado inicial calculado antes de arrancar el reloj.
        self._recalcular_rutas(registrar_cambios=False)
        self._ejecutar_control()

        # --- Procesos SimPy -----------------------------------------------------
        self.env.process(self._llegadas())
        self.env.process(self._periodico(p.intervalo_despacho, self._despachar))
        self.env.process(self._periodico(p.intervalo_rutas, self._recalcular_rutas))
        self.env.process(self._periodico(p.intervalo_control, self._ejecutar_control))
        f = config.fallas
        for nodo in self.topologia.routers()[1:]:  # el gateway no falla
            self.env.process(self._ciclo_fallas(nodo, f.mtbf_router, f.mttr_router))
        for servidor in self.servidores:
            self.env.process(self._ciclo_fallas(servidor, f.mtbf_servidor, f.mttr_servidor))

    # ====================================================================== #
    # Construcción
    # ====================================================================== #
    def _construir_red(self) -> None:
        total = len(self.config.etapas)
        etapas = [[Nodo(i, self._tipo_por_etapa(k, total), k) for i in ids] for k, ids in enumerate(self.config.etapas)]
        enlaces = [Enlace(e.origen, e.destino, e.latencia) for e in self.config.enlaces]

        self.clusters: list[Cluster] = []
        self.servidores: list[Servidor] = []
        for c in self.config.clusters:
            servidores = [Servidor(s.id, total, s.capacidad, c.nombre) for s in c.servidores]
            enlaces += [Enlace(c.switch, s.id, c.latencia_acceso) for s in servidores]
            self.servidores += servidores
            self.clusters.append(Cluster(c.nombre, c.switch, c.eficiencia, c.latencia_acceso, servidores))

        etapas.append(list(self.servidores))  # los servidores forman la última etapa
        self.topologia = Topologia(etapas, enlaces)
        self.cluster_de = {s.id: c for c in self.clusters for s in c.servidores}
        self.capacidad_total = sum(s.capacidad for s in self.servidores)

    def _crear_enrutador(self) -> RutaMasCortaDP | BellmanFordDP:
        """DP por etapas (por defecto) o Bellman-Ford, que admite enlaces entre nodos de una misma etapa."""
        if self.config.algoritmo_rutas == "bellman_ford":
            return BellmanFordDP(list(self.topologia.nodos), self.topologia.gateway.id)
        return RutaMasCortaDP(self.topologia.ids_por_etapa())

    @staticmethod
    def _tipo_por_etapa(k: int, total: int) -> TipoNodo:
        if k == 0:
            return TipoNodo.GATEWAY
        if k == total - 1:
            return TipoNodo.ACCESO
        return TipoNodo.NUCLEO if k == 1 else TipoNodo.AGREGACION

    # ====================================================================== #
    # Control de la ejecución (lo usan el dashboard y la CLI)
    # ====================================================================== #
    @property
    def ahora(self) -> float:
        return self.env.now

    @property
    def duracion(self) -> float:
        return self.config.simulacion.duracion

    @property
    def terminado(self) -> bool:
        return self.env.now >= self.duracion - 1e-9

    def avanzar(self, delta: float) -> None:
        objetivo = min(self.env.now + delta, self.duracion)
        if objetivo > self.env.now:
            self.env.run(until=objetivo)

    def ejecutar(self, al_avanzar: Callable[[float], None] | None = None) -> None:
        """Ejecuta la simulación completa (modo sin interfaz)."""
        tramo = self.duracion / 20
        while not self.terminado:
            self.avanzar(tramo)
            if al_avanzar:
                al_avanzar(self.env.now / self.duracion)

    def ajustar_tasa_llegada(self, delta: float) -> float:
        nueva = round(self.tasa_llegada + delta, 2)
        if not TASA_MINIMA <= nueva <= TASA_MAXIMA:
            raise ValueError(f"La tasa de llegada debe estar entre {TASA_MINIMA} y {TASA_MAXIMA} tareas/s.")
        self.tasa_llegada = nueva
        self._registrar("info", f"Tasa de llegada base ajustada a {nueva:.1f} tareas/s")
        return nueva

    def inyectar_falla(self) -> str:
        """Provoca la falla de un router activo elegido al azar (demostración)."""
        candidatos = [n for n in self.topologia.routers()[1:] if n.activo]
        if not candidatos:
            raise ValueError("No hay routers activos para inyectar una falla.")
        nodo = self.azar.choice(candidatos)
        self._fallar(nodo, manual=True)
        self.env.process(self._reparar_luego(nodo, self.config.fallas.mttr_router))
        return nodo.id

    # Datos instantáneos para el dashboard ----------------------------------
    @property
    def en_proceso(self) -> int:
        return sum(len(s.procesos) for s in self.servidores)

    def tasa_instantanea(self, t: float | None = None) -> float:
        """λ(t) = λ_base · (1 + A·sen(2πt / T)): demanda con hora pico."""
        p = self.config.simulacion
        t = self.env.now if t is None else t
        return self.tasa_llegada * (1 + p.amplitud_pico * math.sin(2 * math.pi * t / p.periodo_pico))

    def ruta_de_cluster(self, indice: int) -> tuple[str, ...] | None:
        return self.rutas_actuales.get(self.clusters[indice].nombre)

    # ====================================================================== #
    # Procesos SimPy
    # ====================================================================== #
    def _periodico(self, intervalo: float, accion: Callable[[], None]):
        while True:
            yield self.env.timeout(intervalo)
            accion()

    def _llegadas(self):
        """Proceso de Poisson no homogéneo mediante el método de thinning.

        Se generan candidatos con la tasa máxima λ_max y cada uno se acepta con
        probabilidad λ(t)/λ_max; el resultado es un proceso de Poisson con tasa λ(t).
        """
        while True:
            tasa_maxima = self.tasa_llegada * (1 + self.config.simulacion.amplitud_pico)
            yield self.env.timeout(self.azar.expovariate(tasa_maxima))
            if self.azar.random() <= self.tasa_instantanea() / tasa_maxima:
                self.cola.append(self._crear_tarea())
                self.metricas.generadas += 1

    def _crear_tarea(self) -> Tarea:
        tipos = self.config.tipos_tarea
        tipo = self.azar.choices(tipos, weights=[t.probabilidad for t in tipos])[0]
        return Tarea(
            id=next(self._ids),
            tipo=tipo,
            cpu=self.azar.randint(tipo.cpu_min, tipo.cpu_max),
            prioridad=self.azar.randint(tipo.prioridad_min, tipo.prioridad_max),
            duracion=self.azar.expovariate(1 / tipo.tiempo_servicio),
            llegada=self.env.now,
        )

    def _despachar(self) -> None:
        """Asigna tareas de la cola a los servidores con el modelo de volumen-carga."""
        ahora = self.env.now
        espera_maxima = self.config.simulacion.espera_maxima
        vigentes = []
        for tarea in self.cola:  # las tareas que superan la espera máxima se descartan
            if tarea.espera(ahora) > espera_maxima:
                self.metricas.registrar_descarte(tarea)
            else:
                vigentes.append(tarea)
        self.cola = vigentes

        # Primero los servidores con más capacidad libre (y alcanzables por la red).
        # Los servidores en emergencia térmica no reciben tareas nuevas (protección térmica).
        candidatos = [
            s for s in self.servidores if s.capacidad_libre > 0 and not s.en_emergencia and self._ruta_a(s)
        ]
        candidatos.sort(key=lambda s: s.capacidad_libre, reverse=True)
        for servidor in candidatos:
            if not self.cola:
                break
            articulos = tuple(Articulo(f"T{t.id}", t.cpu, t.valor(ahora, espera_maxima)) for t in self.cola)
            resultado = self.mochila.resolver(articulos, servidor.capacidad_libre)
            if not resultado.seleccionados:
                continue
            self.metricas.registrar_mochila(resultado.utilizacion)
            self.ultima_mochila = DecisionMochila(
                ahora,
                servidor.id,
                resultado.capacidad,
                articulos,
                resultado.seleccionados,
                resultado.valor_total,
                resultado.peso_total,
            )
            if self._es_mas_ilustrativa(self.ultima_mochila, self.mochila_destacada):
                self.mochila_destacada = self.ultima_mochila
            elegidos = set(resultado.seleccionados)
            ruta = self._ruta_a(servidor)
            for indice in resultado.seleccionados:
                tarea = self.cola[indice]
                servidor.reservar(tarea.cpu)
                self.tareas_en_transito += 1
                self.env.process(self._transmitir(tarea, servidor, ruta))
            self.cola = [t for i, t in enumerate(self.cola) if i not in elegidos]

    @staticmethod
    def _es_mas_ilustrativa(nueva: DecisionMochila, actual: DecisionMochila | None) -> bool:
        """Elige la instancia de la mochila que mejor muestra el modelo en el reporte.

        Se prefieren las instancias (de hasta 15 tareas, para que la tabla quepa) en las que
        el modelo realmente decide: varias tareas caben en la capacidad libre pero no todas
        se pueden tomar a la vez. A igualdad, la que elige más tareas y tiene más candidatas.
        """
        if len(nueva.articulos) > MAX_ARTICULOS_ILUSTRATIVOS:
            return actual is None
        if actual is None or len(actual.articulos) > MAX_ARTICULOS_ILUSTRATIVOS:
            return True

        def puntaje(d: DecisionMochila) -> tuple[bool, int, int]:
            caben = sum(1 for a in d.articulos if a.peso <= d.capacidad)
            decide = len(d.seleccionados) >= 2 and len(d.seleccionados) < caben
            return decide, len(d.seleccionados), caben

        return puntaje(nueva) > puntaje(actual)

    def _ruta_a(self, servidor: Servidor) -> tuple[str, ...] | None:
        """Ruta en uso hacia el switch del clúster más el salto final al servidor."""
        ruta = self.rutas_actuales.get(self.cluster_de[servidor.id].nombre)
        return ruta + (servidor.id,) if ruta else None

    def _transmitir(self, tarea: Tarea, servidor: Servidor, ruta: tuple[str, ...]):
        """Recorre la ruta enlace por enlace; si un nodo cae, el paquete se pierde."""
        factor = self.config.simulacion.factor_congestion
        for origen, destino in zip(ruta, ruta[1:]):
            siguiente = self.topologia.nodos[destino]
            if not siguiente.activo:  # el siguiente salto ya estaba caído
                self._perder_paquete(tarea, servidor)
                return
            enlace = self.topologia.enlace(origen, destino)
            duracion = enlace.tiempo_transito(factor)
            paquete = PaqueteEnTransito(tarea, origen, destino, self.env.now, duracion)
            enlace.en_transito += 1
            enlace.paquetes_totales += 1
            self.paquetes.append(paquete)
            yield self.env.timeout(duracion)
            self.paquetes.remove(paquete)
            enlace.en_transito -= 1
            if not siguiente.activo:  # cayó mientras el paquete viajaba hacia él (incluye el servidor)
                self._perder_paquete(tarea, servidor)
                return

        self.tareas_en_transito -= 1
        servidor.iniciar(tarea.cpu)
        servidor.procesos[tarea.id] = self.env.process(self._procesar(tarea, servidor))

    def _perder_paquete(self, tarea: Tarea, servidor: Servidor) -> None:
        """El paquete no llegó: se libera la reserva y se retransmite desde el gateway."""
        self.tareas_en_transito -= 1
        servidor.cancelar_reserva(tarea.cpu)
        tarea.reintentos += 1
        self.metricas.retransmitidas += 1
        self.cola.append(tarea)

    def _procesar(self, tarea: Tarea, servidor: Servidor):
        cluster = self.cluster_de[servidor.id]
        tarea.inicio_proceso = self.env.now
        valor = tarea.valor(self.env.now, self.config.simulacion.espera_maxima)
        try:
            # La velocidad depende de la energía asignada al clúster (Lagrange).
            yield self.env.timeout(tarea.duracion / cluster.factor_velocidad)
        except simpy.Interrupt:
            # El servidor falló: la tarea vuelve a la cola para reintentarse.
            servidor.liberar(tarea.cpu)
            tarea.reintentos += 1
            self.metricas.interrumpidas += 1
            self.cola.append(tarea)
            return
        servidor.liberar(tarea.cpu)
        servidor.procesos.pop(tarea.id, None)
        servidor.completadas += 1
        self.metricas.registrar_completada(tarea, self.env.now, valor)

    def _ciclo_fallas(self, nodo: Nodo, mtbf: float, mttr: float):
        """Tiempo hasta la falla ~ Exp(1/MTBF); tiempo de reparación ~ Exp(1/MTTR)."""
        while True:
            yield self.env.timeout(self.azar.expovariate(1 / mtbf))
            if not nodo.activo:
                continue  # ya estaba caído (p. ej. por una falla inyectada)
            self._fallar(nodo)
            yield self.env.timeout(self.azar.expovariate(1 / mttr))
            self._reparar(nodo)

    def _reparar_luego(self, nodo: Nodo, mttr: float):
        yield self.env.timeout(self.azar.expovariate(1 / mttr))
        self._reparar(nodo)

    def _fallar(self, nodo: Nodo, manual: bool = False) -> None:
        nodo.fallar()
        self.metricas.fallas += 1
        origen = "inyectada" if manual else "estocástica"
        self._registrar("falla", f"Falla {origen} en {nodo.tipo.value} {nodo.id}")
        if isinstance(nodo, Servidor):
            for proceso in nodo.procesos.values():
                if proceso.is_alive:
                    proceso.interrupt("falla del servidor")
            nodo.procesos.clear()
        else:
            self._recalcular_rutas(actualizar_congestion=False)  # reenrutar de inmediato

    def _reparar(self, nodo: Nodo) -> None:
        if nodo.activo:
            return
        nodo.reparar()
        self._registrar("reparacion", f"{nodo.id} reparado y nuevamente en servicio")
        if not isinstance(nodo, Servidor):
            self._recalcular_rutas(actualizar_congestion=False)

    # ====================================================================== #
    # Módulo 1: enrutamiento con programación dinámica
    # ====================================================================== #
    def _recalcular_rutas(self, registrar_cambios: bool = True, actualizar_congestion: bool = True) -> None:
        """Resuelve la DP con las latencias actuales y actualiza la ruta en uso de cada clúster.

        Para evitar oscilaciones (route flapping) la ruta en uso solo se reemplaza si
        deja de ser válida o si la óptima es al menos UMBRAL_CAMBIO_RUTA más rápida.
        """
        p = self.config.simulacion
        if actualizar_congestion:
            for enlace in self.topologia.enlaces.values():
                enlace.actualizar_congestion(p.suavizado_congestion)
        arcos = self.topologia.arcos_disponibles(p.factor_congestion)
        self.resultado_rutas = self.enrutador.resolver(arcos)

        for cluster in self.clusters:
            ruta = self.resultado_rutas.ruta_hacia(cluster.switch)
            optima = tuple(ruta) if ruta else None
            anterior = self.rutas_actuales.get(cluster.nombre)
            costo_anterior = self._costo_ruta(anterior, arcos)
            costo_optimo = self.resultado_rutas.latencia_hacia(cluster.switch)
            if anterior and optima and costo_anterior <= costo_optimo * (1 + UMBRAL_CAMBIO_RUTA):
                nueva, costo = anterior, costo_anterior  # la mejora no compensa cambiar
            else:
                nueva, costo = optima, costo_optimo
            self.rutas_actuales[cluster.nombre] = nueva
            self.latencias_actuales[cluster.nombre] = costo
            self.rutas_optimas[cluster.nombre] = optima
            self.latencias_optimas[cluster.nombre] = costo_optimo
            if not registrar_cambios or nueva == anterior:
                continue
            if nueva is None:
                motivo = "clúster inalcanzable"
            elif anterior is None:
                motivo = "ruta restablecida"
            elif any(not self.topologia.nodos[n].activo for n in anterior):
                motivo = "falla de nodo"
            else:
                motivo = "congestión"
            self.cambios_ruta.append(CambioRuta(self.env.now, cluster.nombre, anterior, nueva, motivo, costo))
            self.metricas.cambios_ruta += 1
            if nueva is None:
                self._registrar("ruta", f"{cluster.nombre}: sin ruta disponible, clúster inalcanzable")
            else:
                self._registrar("ruta", f"{cluster.nombre}: nueva ruta {' > '.join(nueva)} ({motivo})")

    @staticmethod
    def _costo_ruta(ruta: tuple[str, ...] | None, arcos: dict[tuple[str, str], float]) -> float:
        """Latencia actual de una ruta; infinito si algún enlace ya no está disponible."""
        if not ruta:
            return math.inf
        return sum(arcos.get(tramo, math.inf) for tramo in zip(ruta, ruta[1:]))

    # ====================================================================== #
    # Módulo 2: modelo térmico, método de gradiente y Lagrange
    # ====================================================================== #
    def _ejecutar_control(self) -> None:
        self._actualizar_temperaturas()

        # Método de gradiente: niveles de ventilación que minimizan la energía.
        minimo = self.config.refrigeracion.nivel_minimo
        self.ultimo_gradiente = self.refrigeracion.optimizar(
            cargas=[c.carga for c in self.clusters],
            niveles_iniciales=[c.nivel_ventilador for c in self.clusters],  # arranque en caliente
            minimos=[1.0 if c.en_emergencia else minimo for c in self.clusters],
        )
        for cluster, nivel in zip(self.clusters, self.ultimo_gradiente.niveles):
            cluster.nivel_ventilador = nivel

        # Lagrange: reparto del presupuesto eléctrico entre clústeres.
        # a_i = eficiencia · (capacidad activa / capacidad total) · (carga base + carga actual)
        carga_base = self.config.energia.carga_base
        self.ultimos_coeficientes = tuple(
            c.eficiencia * (c.capacidad_activa / self.capacidad_total) * (carga_base + c.carga) for c in self.clusters
        )
        self.ultimo_lagrange = self.asignador.asignar(self.ultimos_coeficientes)
        presupuesto = self.config.energia.presupuesto_total
        for cluster, energia in zip(self.clusters, self.ultimo_lagrange.energias):
            referencia = presupuesto * cluster.capacidad_total / self.capacidad_total  # reparto proporcional
            cluster.energia_asignada = energia
            # Rendimiento ∝ √E: con más energía (frecuencia de CPU) se procesa más rápido.
            cluster.factor_velocidad = min(max(math.sqrt(energia / referencia), 0.5), 1.5)

        self._tomar_muestra()

    def _actualizar_temperaturas(self) -> None:
        """Temperatura de cada servidor con inercia térmica (sistema de primer orden).

        T(carga) del modelo es la temperatura de equilibrio; la temperatura real se
        acerca a ella con constante de tiempo τ:  T ← T + (T(carga) − T)·(1 − e^(−Δt/τ)).
        Así un pico breve de carga no dispara la emergencia, pero una carga sostenida sí.
        """
        termico = self.config.termico
        alfa = 1 - math.exp(-self.config.simulacion.intervalo_control / termico.inercia)
        for cluster in self.clusters:
            for servidor in cluster.servidores:
                carga = servidor.carga_porcentaje if servidor.activo else 0.0
                objetivo = self.modelo_termico.temperatura(carga)
                servidor.temperatura += (objetivo - servidor.temperatura) * alfa
                umbral = termico.temperatura_emergencia
                if servidor.en_emergencia:  # histéresis: sale al bajar 5 °C del umbral
                    umbral -= HISTERESIS_TERMICA
                servidor.en_emergencia = servidor.activo and servidor.temperatura >= umbral

            cluster.temperatura_max = max(s.temperatura for s in cluster.servidores)
            calientes = [s.id for s in cluster.servidores if s.en_emergencia]
            if calientes and not cluster.en_emergencia:
                self._registrar(
                    "emergencia",
                    f"{cluster.nombre}: {', '.join(calientes)} ≥ {termico.temperatura_emergencia:.0f} °C, "
                    "enfriamiento de emergencia",
                )
            elif cluster.en_emergencia and not calientes:
                self._registrar("info", f"{cluster.nombre}: temperatura normalizada ({cluster.temperatura_max:.1f} °C)")
            cluster.en_emergencia = bool(calientes)

    def _tomar_muestra(self) -> None:
        intervalo = self.config.simulacion.intervalo_control
        completadas = self.metricas.procesadas - self._procesadas_previas
        self._procesadas_previas = self.metricas.procesadas
        self.metricas.muestras.append(
            Muestra(
                t=self.env.now,
                en_cola=len(self.cola),
                en_transito=self.tareas_en_transito,
                en_proceso=self.en_proceso,
                utilizacion=sum(s.en_uso for s in self.servidores) / self.capacidad_total,
                throughput=completadas / intervalo if self.env.now > 0 else 0.0,
                tasa_llegada=self.tasa_instantanea(),
                niveles_ventilador=self.ultimo_gradiente.niveles,
                energia_refrigeracion=self.ultimo_gradiente.energia,
                iteraciones_gradiente=self.ultimo_gradiente.iteraciones,
                energias_cluster=self.ultimo_lagrange.energias,
                multiplicador=self.ultimo_lagrange.multiplicador,
                rendimiento_total=self.ultimo_lagrange.rendimiento_total,
                temperaturas=tuple(c.temperatura_max for c in self.clusters),
            )
        )

    def _registrar(self, tipo: str, mensaje: str) -> None:
        self.metricas.registrar_evento(self.env.now, tipo, mensaje)
