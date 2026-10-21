"""Dashboard en tiempo real con Pygame.

El bucle principal avanza el reloj de SimPy en función del tiempo real
(multiplicado por la velocidad elegida) y redibuja todos los paneles.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from ..reportes.servicio import DIRECTORIO_REPORTES, generar_reporte  # noqa: E402
from ..simulacion.centro_datos import SimuladorCentroDatos  # noqa: E402
from ..simulacion.resumen import construir_resumen  # noqa: E402
from . import tema  # noqa: E402
from .boton import BarraBotones  # noqa: E402
from .graficos import grafico_barras, grafico_lineas  # noqa: E402
from .panel_topologia import PanelTopologia  # noqa: E402

ANCHO, ALTO = 1440, 900  # resolución lógica; la ventana se escala a la pantalla
VELOCIDADES = [0.5, 1, 2, 4, 8, 16, 32]
VENTANA_GRAFICOS = 120.0  # segundos simulados visibles en los gráficos temporales
PASO_TASA = 0.5


class Dashboard:
    def __init__(self, simulador: SimuladorCentroDatos, velocidad: float = 4):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("DataStream Networks · Simulador de centro de datos")
        self.ventana = pygame.display.set_mode(self._tamano_ventana(), pygame.RESIZABLE)
        self.lienzo = pygame.Surface((ANCHO, ALTO))
        self.fuentes = tema.Fuentes()
        self.sim = simulador
        self.indice_velocidad = min(range(len(VELOCIDADES)), key=lambda i: abs(VELOCIDADES[i] - velocidad))
        self.pausado = False
        self.activo = True
        self._mensaje = ("Simulación iniciada · presione [Espacio] para pausar", time.time())
        self._generando_reporte = False
        self._aviso_final = False
        self.topologia = PanelTopologia(simulador, pygame.Rect(12, 68, 868, 540), self.fuentes)
        self.rect_indicadores = pygame.Rect(12, 616, 430, 276)
        self.rect_eventos = pygame.Rect(450, 616, 430, 276)
        self.rect_gradiente = pygame.Rect(890, 68, 538, 270)
        self.rect_lagrange = pygame.Rect(890, 346, 538, 270)
        self.rect_termico = pygame.Rect(890, 624, 538, 268)
        self.barra = self._crear_botones()
        self._mouse: tuple[float, float] | None = None  # posición del mouse en coordenadas del lienzo

    def _crear_botones(self) -> BarraBotones:
        """Botones clicables equivalentes a las teclas (el teclado sigue funcionando igual)."""
        barra = BarraBotones(pygame.Rect(14, 38, 866, 24))
        barra.agregar(lambda: "Reanudar [Espacio]" if self.pausado else "Pausa [Espacio]", self._alternar_pausa)
        barra.agregar("Velocidad − [-]", lambda: self._cambiar_velocidad(-1))
        barra.agregar("Velocidad + [+]", lambda: self._cambiar_velocidad(+1))
        # La fuente por defecto de pygame no tiene flechas (↑↓): se escriben con palabras.
        barra.agregar("Tasa − [Abajo]", lambda: self._cambiar_tasa(-PASO_TASA))
        barra.agregar("Tasa + [Arriba]", lambda: self._cambiar_tasa(PASO_TASA))
        barra.agregar("Falla [F]", self._inyectar_falla)
        barra.agregar("Reporte PDF [R]", self._iniciar_reporte)
        barra.agregar("Salir [Esc]", self._salir)
        barra.distribuir(self.fuentes)
        return barra

    @staticmethod
    def _tamano_ventana() -> tuple[int, int]:
        info = pygame.display.Info()
        if info.current_w <= 0 or info.current_h <= 0:
            return ANCHO, ALTO
        escala = min(1.0, (info.current_w - 40) / ANCHO, (info.current_h - 90) / ALTO)
        return int(ANCHO * escala), int(ALTO * escala)

    @property
    def velocidad(self) -> float:
        return VELOCIDADES[self.indice_velocidad]

    # ================================================================== #
    # Bucle principal
    # ================================================================== #
    def ejecutar(self) -> None:
        reloj = pygame.time.Clock()
        while self.activo:
            dt = min(reloj.tick(60) / 1000, 0.1)  # evita saltos si la ventana se congela
            self._procesar_eventos()
            self.actualizar(dt)
            self.dibujar()
            self._presentar()
        pygame.quit()

    def actualizar(self, dt: float) -> None:
        if not self.pausado and not self.sim.terminado:
            self.sim.avanzar(dt * self.velocidad)
        if self.sim.terminado and not self._aviso_final:
            self._aviso_final = True
            self.notificar("Simulación finalizada · presione [R] para generar el reporte PDF")
        self.topologia.actualizar(time.time())

    def _presentar(self) -> None:
        if self.ventana.get_size() == self.lienzo.get_size():
            self.ventana.blit(self.lienzo, (0, 0))
        else:
            self.ventana.blit(pygame.transform.smoothscale(self.lienzo, self.ventana.get_size()), (0, 0))
        pygame.display.flip()

    def notificar(self, mensaje: str) -> None:
        self._mensaje = (mensaje, time.time())

    # ================================================================== #
    # Comandos del usuario (todos validados)
    # ================================================================== #
    def _procesar_eventos(self) -> None:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.activo = False
            elif evento.type == pygame.KEYDOWN:
                self._procesar_tecla(evento)
            elif evento.type == pygame.MOUSEMOTION:
                self._mouse = self._a_lienzo(evento.pos)
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:  # clic izquierdo
                self._procesar_clic(self._a_lienzo(evento.pos))

    def _a_lienzo(self, posicion: tuple[int, int]) -> tuple[float, float]:
        """Convierte una posición de la ventana (que puede estar escalada) a coordenadas del lienzo."""
        ancho, alto = self.ventana.get_size()
        return posicion[0] * ANCHO / ancho, posicion[1] * ALTO / alto

    def _procesar_clic(self, punto: tuple[float, float]) -> None:
        boton = self.barra.boton_en(punto)
        if boton is not None:
            boton.pulsar()
            return
        indice = self.topologia.cluster_en(punto)  # clic sobre el recuadro de un clúster
        if indice is not None:
            self._seleccionar_cluster(indice)

    def _procesar_tecla(self, evento: pygame.event.Event) -> None:
        tecla = evento.key
        if tecla in (pygame.K_ESCAPE, pygame.K_q):
            self._salir()
        elif tecla == pygame.K_SPACE:
            self._alternar_pausa()
        elif tecla in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._cambiar_velocidad(+1)
        elif tecla in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._cambiar_velocidad(-1)
        elif tecla in (pygame.K_UP, pygame.K_DOWN):
            self._cambiar_tasa(PASO_TASA if tecla == pygame.K_UP else -PASO_TASA)
        elif pygame.K_1 <= tecla <= pygame.K_9:
            self._seleccionar_cluster(tecla - pygame.K_1)
        elif tecla == pygame.K_f:
            self._inyectar_falla()
        elif tecla == pygame.K_r:
            self._iniciar_reporte()
        elif evento.unicode and evento.unicode.isprintable():
            self.notificar(f"Comando '{evento.unicode}' no reconocido · use los botones o las teclas indicadas")

    def _salir(self) -> None:
        self.activo = False

    def _alternar_pausa(self) -> None:
        if self.sim.terminado:
            self.notificar("La simulación ya terminó; no se puede reanudar")
            return
        self.pausado = not self.pausado
        self.notificar("Simulación en pausa" if self.pausado else "Simulación reanudada")

    def _seleccionar_cluster(self, indice: int) -> None:
        if self.topologia.seleccionar_cluster(indice):
            self.notificar(f"Mostrando la ruta óptima hacia {self.sim.clusters[indice].nombre}")
        else:
            self.notificar(f"No existe el clúster {indice + 1} (hay {len(self.sim.clusters)})")

    def _cambiar_velocidad(self, direccion: int) -> None:
        nuevo = self.indice_velocidad + direccion
        if not 0 <= nuevo < len(VELOCIDADES):
            limite = "máxima" if direccion > 0 else "mínima"
            self.notificar(f"Velocidad {limite} alcanzada (×{self.velocidad:g})")
            return
        self.indice_velocidad = nuevo
        self.notificar(f"Velocidad de simulación ×{self.velocidad:g}")

    def _cambiar_tasa(self, delta: float) -> None:
        try:
            nueva = self.sim.ajustar_tasa_llegada(delta)
            self.notificar(f"Tasa de llegada base: {nueva:.1f} tareas/s")
        except ValueError as error:
            self.notificar(str(error))

    def _inyectar_falla(self) -> None:
        if self.sim.terminado:
            self.notificar("La simulación terminó; no se pueden inyectar fallas")
            return
        try:
            nodo = self.sim.inyectar_falla()
            self.notificar(f"Falla inyectada en {nodo}: observe el cambio de ruta")
        except ValueError as error:
            self.notificar(str(error))

    def _iniciar_reporte(self) -> None:
        if self._generando_reporte:
            self.notificar("Ya se está generando un reporte, espere un momento")
            return
        if self.sim.ahora < 10:
            self.notificar("Deje correr la simulación al menos 10 s antes de generar el reporte")
            return
        DIRECTORIO_REPORTES.mkdir(parents=True, exist_ok=True)
        captura = DIRECTORIO_REPORTES / f"captura_{datetime.now():%Y%m%d_%H%M%S}.png"
        pygame.image.save(self.lienzo, str(captura))
        resumen = construir_resumen(self.sim)  # copia del estado: el hilo no toca la simulación
        self._generando_reporte = True
        self.notificar("Generando reporte PDF con análisis de IA...")
        threading.Thread(target=self._generar_reporte, args=(resumen, captura), daemon=True).start()

    def _generar_reporte(self, resumen: dict, captura: Path) -> None:
        try:
            resultado = generar_reporte(resumen, captura)
            self.notificar(f"Reporte guardado: {resultado.ruta.name} (análisis: {resultado.fuente_analisis})")
        except Exception as error:  # el hilo nunca debe tumbar el dashboard
            self.notificar(f"No se pudo generar el reporte: {error}")
        finally:
            self._generando_reporte = False

    # ================================================================== #
    # Dibujo
    # ================================================================== #
    def dibujar(self) -> None:
        self.lienzo.fill(tema.FONDO)
        ahora_real = time.time()
        self._dibujar_encabezado()
        self.topologia.dibujar(self.lienzo, ahora_real)
        self._dibujar_indicadores()
        self._dibujar_eventos()
        self._dibujar_gradiente()
        self._dibujar_lagrange()
        self._dibujar_termico()
        self._dibujar_mensaje(ahora_real)

    def _dibujar_encabezado(self) -> None:
        f = self.fuentes
        f.texto(self.lienzo, "DataStream Networks · Simulador de optimización del centro de datos", (14, 10), 28, tema.TEXTO, True)
        self.barra.dibujar(self.lienzo, f, self._mouse)
        if self.sim.terminado:
            estado, color = "FINALIZADA", tema.AZUL
        elif self.pausado:
            estado, color = "EN PAUSA", tema.AMARILLO
        else:
            estado, color = "EN EJECUCIÓN", tema.VERDE
        f.texto(self.lienzo, estado, (ANCHO - 14, 10), 22, color, True, "derecha")
        f.texto(
            self.lienzo,
            f"t = {self.sim.ahora:6.1f} / {self.sim.duracion:.0f} s   ·   ×{self.velocidad:g}   ·   "
            f"λ(t) = {self.sim.tasa_instantanea():.2f} tareas/s (base {self.sim.tasa_llegada:.1f})",
            (ANCHO - 14, 34),
            17,
            tema.TEXTO,
            alineacion="derecha",
        )
        progreso = pygame.Rect(890, 54, 538, 4)
        pygame.draw.rect(self.lienzo, tema.BORDE, progreso)
        pygame.draw.rect(self.lienzo, color, (progreso.x, progreso.y, progreso.width * self.sim.ahora / self.sim.duracion, 4))

    def _dibujar_indicadores(self) -> None:
        area = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_indicadores, "Indicadores (teoría de colas)")
        m = self.sim.metricas
        ultima = m.muestras[-1] if m.muestras else None
        recientes = m.muestras[-10:]  # throughput promediado en los últimos 10 ciclos
        throughput = sum(x.throughput for x in recientes) / len(recientes) if recientes else 0.0
        caidos = sum(1 for n in self.sim.topologia.nodos.values() if not n.activo)
        tarjetas = [
            ("Generadas", f"{m.generadas}", tema.TEXTO),
            ("Procesadas", f"{m.procesadas}", tema.VERDE),
            ("Descartadas", f"{m.descartadas}", tema.ROJO if m.descartadas else tema.TEXTO),
            ("Retransmitidas", f"{m.retransmitidas}", tema.AMARILLO if m.retransmitidas else tema.TEXTO),
            ("En cola", f"{len(self.sim.cola)}", tema.TEXTO),
            ("En tránsito", f"{self.sim.tareas_en_transito}", tema.TEXTO),
            ("En proceso", f"{self.sim.en_proceso}", tema.TEXTO),
            ("Throughput", f"{throughput:.2f} t/s", tema.CIAN),
            ("Espera prom.", f"{m.espera_promedio:.2f} s", tema.TEXTO),
            ("Utilización", f"{ultima.utilizacion:.0%}" if ultima else "-", tema.TEXTO),
            ("Cambios ruta", f"{m.cambios_ruta}", tema.AMARILLO),
            ("Nodos caídos", f"{caidos}", tema.ROJO if caidos else tema.VERDE),
        ]
        ancho, alto = area.width / 4, 58
        for i, (etiqueta, valor, color) in enumerate(tarjetas):
            x, y = area.x + (i % 4) * ancho, area.y + (i // 4) * (alto + 6)
            tarjeta = pygame.Rect(x + 3, y, ancho - 6, alto)
            pygame.draw.rect(self.lienzo, (30, 36, 44), tarjeta, border_radius=6)
            self.fuentes.texto(self.lienzo, etiqueta, (tarjeta.x + 8, tarjeta.y + 7), 15, tema.TEXTO_SUAVE)
            self.fuentes.texto(self.lienzo, valor, (tarjeta.x + 8, tarjeta.y + 27), 26, color, True)

        decision = self.sim.ultima_mochila
        y = area.bottom - 30
        if decision:
            self.fuentes.texto(
                self.lienzo,
                f"Última mochila (volumen-carga) en {decision.servidor}: capacidad {decision.capacidad} · "
                f"{len(decision.seleccionados)} de {len(decision.articulos)} tareas",
                (area.x + 4, y),
                15,
                tema.TEXTO_SUAVE,
            )
            self.fuentes.texto(
                self.lienzo,
                f"valor f(W) = {decision.valor_total:.1f} · peso {decision.peso_total}/{decision.capacidad} · "
                f"utilización media {m.utilizacion_mochila_promedio:.0%}",
                (area.x + 4, y + 16),
                15,
                tema.TEXTO_SUAVE,
            )

    def _dibujar_eventos(self) -> None:
        area = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_eventos, "Bitácora de eventos")
        fuente = self.fuentes.obtener(15)
        eventos = list(self.sim.metricas.eventos)[-14:]
        for i, evento in enumerate(reversed(eventos)):
            texto = f"[{evento.t:6.1f} s] {evento.mensaje}"
            while fuente.size(texto)[0] > area.width - 8 and len(texto) > 10:
                texto = texto[:-2]
            color = tema.COLORES_EVENTO.get(evento.tipo, tema.TEXTO)
            self.fuentes.texto(self.lienzo, texto, (area.x + 4, area.y + i * 17), 15, color)

    def _series_por_cluster(self, atributo: str, desde: float, escala: float = 1.0):
        tiempos, valores = self.sim.metricas.serie(atributo, desde)
        series = []
        for i, cluster in enumerate(self.sim.clusters):
            color = tema.COLORES_CLUSTER[i % len(tema.COLORES_CLUSTER)]
            series.append((cluster.nombre.split()[-1], color, [v[i] * escala for v in valores]))
        return tiempos, series

    def _dibujar_gradiente(self) -> None:
        area = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_gradiente, "Método de gradiente · refrigeración")
        resultado = self.sim.ultimo_gradiente
        self.fuentes.texto(
            self.lienzo,
            f"min E(u) = {resultado.energia:.2f} kW · {resultado.iteraciones} iteraciones · "
            f"norma del gradiente proyectado = {resultado.norma_gradiente:.1e}",
            (area.x + 2, area.y),
            16,
            tema.CIAN,
        )
        desde = max(0.0, self.sim.ahora - VENTANA_GRAFICOS)
        tiempos, series = self._series_por_cluster("niveles_ventilador", desde, 100)
        izquierda = pygame.Rect(area.x, area.y + 20, int(area.width * 0.62), area.height - 20)
        grafico_lineas(
            self.lienzo, self.fuentes, izquierda, tiempos, series, 0, 100,
            formato_y=lambda v: f"{v:.0f}%", titulo="Potencia de ventiladores u_i(t)", x_desde=desde,
        )
        historial = resultado.historial_energia
        derecha = pygame.Rect(izquierda.right + 6, izquierda.y, area.right - izquierda.right - 6, izquierda.height)
        y_min, y_max = min(historial), max(historial)
        margen = max((y_max - y_min) * 0.1, 0.05)
        grafico_lineas(
            self.lienzo, self.fuentes, derecha, list(range(len(historial))),
            [("E(k)", tema.AMARILLO, historial)], y_min - margen, y_max + margen,
            formato_y=lambda v: f"{v:.2f}" if y_max - y_min < 1 else f"{v:.1f}",
            titulo="Convergencia: E(u^k) vs k", x_etiqueta=lambda k: f"k={k:.0f}",
        )

    def _dibujar_lagrange(self) -> None:
        area = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_lagrange, "Multiplicadores de Lagrange · presupuesto eléctrico")
        resultado = self.sim.ultimo_lagrange
        self.fuentes.texto(
            self.lienzo,
            f"λ* = {resultado.multiplicador:.4f} rend./kW · Σ E_i = {sum(resultado.energias):.1f} / "
            f"{resultado.presupuesto:.0f} kW · R total = {resultado.rendimiento_total:.2f}",
            (area.x + 2, area.y),
            16,
            tema.CIAN,
        )
        izquierda = pygame.Rect(area.x, area.y + 40, int(area.width * 0.5), area.height - 40)
        colores = [tema.COLORES_CLUSTER[i % len(tema.COLORES_CLUSTER)] for i in range(len(self.sim.clusters))]
        grafico_barras(
            self.lienzo, self.fuentes, izquierda,
            [c.nombre.split()[-1] for c in self.sim.clusters], resultado.energias, colores,
            valor_maximo=max(resultado.energias + (1e-6,)) * 1.15, formato=lambda v: f"{v:.1f} kW",
            subetiquetas=[f"a={a:.2f}" for a in self.sim.ultimos_coeficientes],
        )
        self.fuentes.texto(self.lienzo, "Energía asignada E_i*", (izquierda.x + 2, area.y + 22), 15, tema.TEXTO_SUAVE)
        desde = max(0.0, self.sim.ahora - VENTANA_GRAFICOS)
        tiempos, valores = self.sim.metricas.serie("multiplicador", desde)
        derecha = pygame.Rect(izquierda.right + 6, area.y + 22, area.right - izquierda.right - 6, area.height - 22)
        y_max = max(valores, default=0.05) * 1.2 or 0.05
        grafico_lineas(
            self.lienzo, self.fuentes, derecha, tiempos, [("λ*", tema.AMARILLO, valores)], 0, y_max,
            formato_y=lambda v: f"{v:.3f}", titulo="Precio sombra λ*(t)", x_desde=desde,
        )

    def _dibujar_termico(self) -> None:
        area = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_termico, "Modelo térmico T(x) · puntos estacionarios")
        modelo, analisis = self.sim.modelo_termico, self.sim.analisis_termico
        x_min, x_max = modelo.dominio
        muestras = [x_min + (x_max - x_min) * i / 120 for i in range(121)]
        temperaturas = [modelo.temperatura(v) for v in muestras]
        t_min, t_max = min(temperaturas) - 5, max(temperaturas + [analisis.temperatura_emergencia]) + 5
        zona = pygame.Rect(area.x + 40, area.y + 6, int(area.width * 0.62) - 44, area.height - 26)

        def punto(carga: float, temperatura: float) -> tuple[int, int]:
            px = zona.left + (carga - x_min) / (x_max - x_min) * zona.width
            py = zona.bottom - (min(max(temperatura, t_min), t_max) - t_min) / (t_max - t_min) * zona.height
            return int(px), int(py)

        for i in range(5):
            valor = t_min + (t_max - t_min) * i / 4
            y = punto(x_min, valor)[1]
            pygame.draw.line(self.lienzo, tema.REJILLA, (zona.left, y), (zona.right, y))
            self.fuentes.texto(self.lienzo, f"{valor:.0f}°", (zona.left - 5, y - 6), 14, tema.TEXTO_SUAVE, alineacion="derecha")
        pygame.draw.rect(self.lienzo, tema.BORDE, zona, 1)
        self.fuentes.texto(self.lienzo, f"{x_min:.0f}%", (zona.left, zona.bottom + 3), 14, tema.TEXTO_SUAVE)
        self.fuentes.texto(self.lienzo, "carga x", (zona.centerx, zona.bottom + 3), 14, tema.TEXTO_SUAVE, alineacion="centro")
        self.fuentes.texto(self.lienzo, f"{x_max:.0f}%", (zona.right, zona.bottom + 3), 14, tema.TEXTO_SUAVE, alineacion="derecha")

        # Umbral de emergencia
        y_em = punto(x_min, analisis.temperatura_emergencia)[1]
        tema.linea_discontinua(self.lienzo, tema.ROJO, (zona.left, y_em), (zona.right, y_em), 1)
        if analisis.carga_emergencia is not None:
            x_em = punto(analisis.carga_emergencia, t_min)[0]
            tema.linea_discontinua(self.lienzo, tema.ROJO, (x_em, zona.top), (x_em, zona.bottom), 1)
            pygame.draw.rect(self.lienzo, tema.mezclar(tema.PANEL, tema.ROJO, 0.18), (x_em, zona.top + 1, zona.right - x_em - 1, zona.height - 2))

        pygame.draw.lines(self.lienzo, tema.TEXTO, False, [punto(c, t) for c, t in zip(muestras, temperaturas)], 2)

        # Puntos notables obtenidos con derivadas
        for p in analisis.puntos_estacionarios:
            centro = punto(p.carga, p.temperatura)
            if p.tipo == "máximo local":
                tema.triangulo(self.lienzo, tema.NARANJA, centro, 7, hacia_arriba=True)
            elif p.tipo == "mínimo local":
                tema.triangulo(self.lienzo, tema.VERDE, centro, 7, hacia_arriba=False)
            else:
                tema.rombo(self.lienzo, tema.AMARILLO, centro, 6)
        for p in analisis.puntos_inflexion:
            tema.rombo(self.lienzo, tema.AMARILLO, punto(p.carga, p.temperatura), 6)

        # Servidores: (carga actual, temperatura actual) coloreados por clúster
        for i, cluster in enumerate(self.sim.clusters):
            color = tema.COLORES_CLUSTER[i % len(tema.COLORES_CLUSTER)]
            for servidor in cluster.servidores:
                if servidor.activo:
                    pygame.draw.circle(self.lienzo, color, punto(servidor.carga_porcentaje, servidor.temperatura), 5)

        # Leyenda con los resultados analíticos
        x, y = zona.right + 12, area.y + 4
        self.fuentes.texto(self.lienzo, "T'(x) = 0, criterio de T''(x):", (x, y), 15, tema.TEXTO_SUAVE)
        y += 20
        for p in analisis.puntos_estacionarios:
            color = tema.NARANJA if p.tipo == "máximo local" else tema.VERDE
            if p.tipo == "máximo local":
                tema.triangulo(self.lienzo, color, (x + 6, y + 7), 6, True)
            else:
                tema.triangulo(self.lienzo, color, (x + 6, y + 7), 6, False)
            self.fuentes.texto(self.lienzo, f"{p.tipo}: {p.carga:.1f}%, {p.temperatura:.1f}°C", (x + 16, y), 15, color)
            y += 19
        for p in analisis.puntos_inflexion:
            tema.rombo(self.lienzo, tema.AMARILLO, (x + 6, y + 7), 5)
            self.fuentes.texto(self.lienzo, f"inflexión: {p.carga:.1f}%, {p.temperatura:.1f}°C", (x + 16, y), 15, tema.AMARILLO)
            y += 19
        y += 6
        if analisis.carga_emergencia is not None:
            self.fuentes.texto(self.lienzo, "Punto crítico (emergencia):", (x, y), 15, tema.ROJO, True)
            self.fuentes.texto(
                self.lienzo, f"x_e = {analisis.carga_emergencia:.1f}%  (T ≥ {analisis.temperatura_emergencia:.0f}°C)",
                (x, y + 18), 15, tema.ROJO,
            )
            y += 40
        if analisis.zona_estable_hasta is not None:
            self.fuentes.texto(self.lienzo, f"Zona estable hasta {analisis.zona_estable_hasta:.0f}% (mín. local)", (x, y), 14, tema.TEXTO_SUAVE)
            y += 18
        emergencias = [c.nombre.split()[-1] for c in self.sim.clusters if c.en_emergencia]
        texto = f"En emergencia: {', '.join(emergencias)}" if emergencias else "Sin emergencias térmicas"
        self.fuentes.texto(self.lienzo, texto, (x, area.bottom - 16), 15, tema.ROJO if emergencias else tema.VERDE, True)

    def _dibujar_mensaje(self, ahora_real: float) -> None:
        texto, instante = self._mensaje
        if ahora_real - instante > 4 and not self._generando_reporte:
            return
        if self._generando_reporte:
            texto = "Generando reporte PDF con análisis de IA..."
        fuente = self.fuentes.obtener(18, True)
        ancho = fuente.size(texto)[0] + 28
        caja = pygame.Rect(0, 0, ancho, 32)
        caja.midtop = (self.topologia.rect.centerx, self.topologia.rect.y + 86)
        pygame.draw.rect(self.lienzo, (40, 46, 56), caja, border_radius=8)
        pygame.draw.rect(self.lienzo, tema.AZUL, caja, 1, border_radius=8)
        self.fuentes.texto(self.lienzo, texto, (caja.centerx, caja.y + 8), 18, tema.TEXTO, True, "centro")
