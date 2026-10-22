"""Interfaz gráfica (Pygame) del método Simplex.

Muestra cada tabla del Simplex con la variable que entra, la que sale y el pivote
resaltados, la explicación de cada paso, la formulación y el resultado final. Con
dos variables dibuja la región factible, las restricciones, la recta de Z y el
recorrido del Simplex de vértice en vértice; con más variables muestra barras con
la solución básica y el uso de cada restricción en la tabla actual.
"""

from __future__ import annotations

import math
import os
import time
from typing import Sequence

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from ..consola import expresion_lineal, fraccion  # noqa: E402
from ..modelos.programacion_lineal import (  # noqa: E402
    ProblemaLineal,
    ResolvedorSimplex,
    ResultadoSimplex,
    TablaSimplex,
    resolver_con_scipy,
)
from . import tema  # noqa: E402
from .boton import BarraBotones  # noqa: E402
from .graficos import grafico_barras, grafico_lineas  # noqa: E402

ANCHO, ALTO = 1440, 900
SIGNOS = {"<=": "≤", ">=": "≥", "=": "="}
ETIQUETAS_EJEMPLO = {
    "centro-datos": "Centro de datos",
    "parcial-iv": "Parcial IV",
    "minimizar": "Minimizar",
    "no-acotado": "No acotado",
    "infactible": "Infactible",
    "propio": "Propio",
}
COLORES_RESTRICCION = tema.COLORES_TAREA


def valores_de_tabla(tabla: TablaSimplex, nombres: Sequence[str]) -> tuple[float, ...]:
    """Solución básica de una tabla: las variables básicas valen su b_i y las no básicas 0."""
    valores = dict(zip(nombres, [0.0] * len(nombres)))
    for variable, fila in zip(tabla.base, tabla.filas):
        if variable in valores:
            valores[variable] = fila[-1]
    return tuple(valores[x] for x in nombres)


def valor_objetivo_de_tabla(tabla: TablaSimplex) -> float:
    """Valor de la fase en la tabla: W en la fase 1, Z en la fase 2 (con el signo original si se minimiza)."""
    return -tabla.fila_objetivo[-1] if tabla.etiqueta_objetivo == "-Z" else tabla.fila_objetivo[-1]


def direccion_no_acotada(tabla: TablaSimplex, nombres: Sequence[str]) -> tuple[float, ...] | None:
    """Dirección en que se mueven las variables de decisión cuando la que entra no tiene razón mínima.

    Al aumentar la entrante en t, cada básica cambia en −a_ik·t; si la entrante es una
    variable de decisión, ella misma aumenta en t.
    """
    if tabla.entrante is None or tabla.saliente is not None:
        return None
    k = tabla.columnas.index(tabla.entrante)
    cambio = {variable: -fila[k] for variable, fila in zip(tabla.base, tabla.filas)}
    cambio[tabla.entrante] = 1.0
    direccion = tuple(cambio.get(x, 0.0) for x in nombres)
    return direccion if any(abs(d) > 1e-12 for d in direccion) else None


def _recortar(poligono: list[tuple[float, float]], a1: float, a2: float, b: float) -> list[tuple[float, float]]:
    """Sutherland-Hodgman: parte del polígono que cumple a1·x + a2·y <= b."""
    resultado = []
    for i, p in enumerate(poligono):
        q = poligono[(i + 1) % len(poligono)]
        valor_p, valor_q = a1 * p[0] + a2 * p[1] - b, a1 * q[0] + a2 * q[1] - b
        if valor_p <= 1e-9:
            resultado.append(p)
        if (valor_p <= 1e-9) != (valor_q <= 1e-9):
            t = valor_p / (valor_p - valor_q)
            resultado.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return resultado


def region_factible(problema: ProblemaLineal, ancho: float, alto: float) -> list[tuple[float, float]]:
    """Polígono de la región factible de un problema de 2 variables, recortado a [0, ancho] × [0, alto]."""
    poligono = [(0.0, 0.0), (ancho, 0.0), (ancho, alto), (0.0, alto)]
    for r in problema.restricciones:
        a1, a2 = r.coeficientes
        if r.sentido in ("<=", "="):
            poligono = _recortar(poligono, a1, a2, r.lado_derecho)
        if r.sentido in (">=", "="):
            poligono = _recortar(poligono, -a1, -a2, -r.lado_derecho)
        if not poligono:
            return []
    return poligono


def segmento_en_caja(a1: float, a2: float, b: float, ancho: float, alto: float):
    """Tramo de la recta a1·x + a2·y = b dentro de [0, ancho] × [0, alto], o None."""
    puntos = []
    if a2:
        puntos += [(x, (b - a1 * x) / a2) for x in (0.0, ancho)]
    if a1:
        puntos += [((b - a2 * y) / a1, y) for y in (0.0, alto)]
    dentro = [(x, y) for x, y in puntos if -1e-9 <= x <= ancho + 1e-9 and -1e-9 <= y <= alto + 1e-9]
    if len(dentro) < 2:
        return None
    dentro.sort()
    return dentro[0], dentro[-1]


def limites_grafico(problema: ProblemaLineal, puntos: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Ancho y alto de la vista: abarca los cortes con los ejes, los vértices y los puntos del recorrido."""
    xs, ys = [p[0] for p in puntos], [p[1] for p in puntos]
    restricciones = problema.restricciones
    for r in restricciones:
        (a1, a2), b = r.coeficientes, r.lado_derecho
        if a1 and b / a1 > 0:
            xs.append(b / a1)
        if a2 and b / a2 > 0:
            ys.append(b / a2)
    for i, r in enumerate(restricciones):
        for s in restricciones[i + 1:]:
            (a, b), (c, d) = r.coeficientes, s.coeficientes
            determinante = a * d - b * c
            if abs(determinante) > 1e-12:
                x = (r.lado_derecho * d - b * s.lado_derecho) / determinante
                y = (a * s.lado_derecho - r.lado_derecho * c) / determinante
                if x >= -1e-9 and y >= -1e-9:
                    xs.append(x)
                    ys.append(y)
    ancho, alto = max(xs, default=0.0), max(ys, default=0.0)
    ancho, alto = ancho or alto or 1.0, alto or ancho or 1.0
    return ancho * 1.2, alto * 1.2


class VentanaLineal:
    """Ventana para recorrer el Simplex tabla por tabla y cambiar de problema."""

    def __init__(self, problemas: dict[str, tuple[ProblemaLineal, str]], inicial: str):
        if inicial not in problemas:
            raise ValueError(f"Problema '{inicial}' desconocido. Opciones: {', '.join(problemas)}")
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("DataStream Networks · Programación lineal (método Simplex)")
        self.ventana = pygame.display.set_mode(self._tamano_ventana(), pygame.RESIZABLE)
        self.lienzo = pygame.Surface((ANCHO, ALTO))
        self.fuentes = tema.Fuentes()
        self.problemas = problemas
        self.activo = True
        self._mouse: tuple[float, float] | None = None
        self._mensaje = ("Use [Izq] y [Der] (o los botones) para recorrer las tablas", time.time())
        self.rect_tabla = pygame.Rect(12, 68, 868, 470)
        self.rect_problema = pygame.Rect(12, 546, 430, 346)
        self.rect_resultado = pygame.Rect(450, 546, 430, 346)
        self.rect_grafico = pygame.Rect(890, 68, 538, 824)
        self.barra_ejemplos = BarraBotones(pygame.Rect(14, 38, 866, 24))
        for i, nombre in enumerate(problemas, start=1):
            self.barra_ejemplos.agregar(f"{ETIQUETAS_EJEMPLO.get(nombre, nombre)} [{i}]", lambda n=nombre: self.seleccionar(n))
        self.barra_ejemplos.distribuir(self.fuentes)
        self.barra_pasos = BarraBotones(pygame.Rect(890, 38, 538, 24))
        self.barra_pasos.agregar("< Anterior [Izq]", lambda: self.mover(-1))
        self.barra_pasos.agregar("Siguiente > [Der]", lambda: self.mover(+1))
        self.barra_pasos.agregar("Inicio", lambda: self.ir_a(0))
        self.barra_pasos.agregar("Final", lambda: self.ir_a(len(self.pasos) - 1))
        self.barra_pasos.agregar("Salir [Esc]", self._salir)
        self.barra_pasos.distribuir(self.fuentes)
        self.seleccionar(inicial)

    @staticmethod
    def _tamano_ventana() -> tuple[int, int]:
        info = pygame.display.Info()
        if info.current_w <= 0 or info.current_h <= 0:
            return ANCHO, ALTO
        escala = min(1.0, (info.current_w - 40) / ANCHO, (info.current_h - 90) / ALTO)
        return int(ANCHO * escala), int(ALTO * escala)

    def seleccionar(self, nombre: str) -> None:
        self.nombre = nombre
        self.problema, self.descripcion = self.problemas[nombre]
        self.resultado: ResultadoSimplex = ResolvedorSimplex().resolver(self.problema)
        self.verificacion = resolver_con_scipy(self.problema)
        self.pasos = list(self.resultado.tablas)
        if self.resultado.tabla_alternativa is not None:
            self.pasos.append(self.resultado.tabla_alternativa)
        self.paso = 0
        self.notificar(f"{ETIQUETAS_EJEMPLO.get(nombre, nombre)}: {len(self.pasos)} tablas")

    def mover(self, delta: int) -> None:
        nuevo = self.paso + delta
        if not 0 <= nuevo < len(self.pasos):
            self.notificar("Ya está en la primera tabla" if nuevo < 0 else "Ya está en la última tabla")
            return
        self.paso = nuevo

    def ir_a(self, paso: int) -> None:
        self.paso = min(max(paso, 0), len(self.pasos) - 1)

    @property
    def es_alternativa(self) -> bool:
        return self.resultado.tabla_alternativa is not None and self.paso == len(self.pasos) - 1

    def notificar(self, mensaje: str) -> None:
        self._mensaje = (mensaje, time.time())

    def _salir(self) -> None:
        self.activo = False

    def ejecutar(self) -> None:
        reloj = pygame.time.Clock()
        while self.activo:
            reloj.tick(30)
            self._procesar_eventos()
            self.dibujar()
            self._presentar()
        pygame.quit()

    def _presentar(self) -> None:
        if self.ventana.get_size() == self.lienzo.get_size():
            self.ventana.blit(self.lienzo, (0, 0))
        else:
            self.ventana.blit(pygame.transform.smoothscale(self.lienzo, self.ventana.get_size()), (0, 0))
        pygame.display.flip()

    def _a_lienzo(self, posicion: tuple[int, int]) -> tuple[float, float]:
        ancho, alto = self.ventana.get_size()
        return posicion[0] * ANCHO / ancho, posicion[1] * ALTO / alto

    def _procesar_eventos(self) -> None:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.activo = False
            elif evento.type == pygame.KEYDOWN:
                self._procesar_tecla(evento)
            elif evento.type == pygame.MOUSEMOTION:
                self._mouse = self._a_lienzo(evento.pos)
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                punto = self._a_lienzo(evento.pos)
                boton = self.barra_ejemplos.boton_en(punto) or self.barra_pasos.boton_en(punto)
                if boton is not None:
                    boton.pulsar()

    def _procesar_tecla(self, evento: pygame.event.Event) -> None:
        tecla = evento.key
        nombres = list(self.problemas)
        if tecla in (pygame.K_ESCAPE, pygame.K_q):
            self._salir()
        elif tecla in (pygame.K_RIGHT, pygame.K_SPACE):
            self.mover(+1)
        elif tecla == pygame.K_LEFT:
            self.mover(-1)
        elif tecla == pygame.K_HOME:
            self.ir_a(0)
        elif tecla == pygame.K_END:
            self.ir_a(len(self.pasos) - 1)
        elif pygame.K_1 <= tecla <= pygame.K_9 and tecla - pygame.K_1 < len(nombres):
            self.seleccionar(nombres[tecla - pygame.K_1])
        elif evento.unicode and evento.unicode.isprintable():
            self.notificar(f"Comando '{evento.unicode}' no reconocido · use los botones o las teclas indicadas")

    def dibujar(self) -> None:
        self.lienzo.fill(tema.FONDO)
        self._dibujar_encabezado()
        self._dibujar_tabla()
        self._dibujar_problema()
        self._dibujar_resultado()
        if len(self.problema.objetivo) == 2:
            self._dibujar_region()
        else:
            self._dibujar_barras()
        self._dibujar_mensaje()

    def _dibujar_encabezado(self) -> None:
        f = self.fuentes
        f.texto(self.lienzo, "Programación lineal · Método Simplex", (14, 10), 28, tema.TEXTO, True)
        f.texto(self.lienzo, self.descripcion, (ANCHO - 14, 12), 18, tema.TEXTO_SUAVE, alineacion="derecha")
        self.barra_ejemplos.dibujar(self.lienzo, f, self._mouse)
        seleccionado = self.barra_ejemplos.botones[list(self.problemas).index(self.nombre)]
        pygame.draw.rect(self.lienzo, tema.VERDE, seleccionado.rect, 2, border_radius=5)
        self.barra_pasos.dibujar(self.lienzo, f, self._mouse)

    def _titulo_tabla(self, tabla: TablaSimplex) -> str:
        numero = f"paso {self.paso + 1} de {len(self.pasos)}"
        if self.es_alternativa:
            return f"Tabla alternativa (soluciones múltiples) · {numero}"
        fase = f"Fase {tabla.fase} · " if self.resultado.usa_dos_fases else ""
        return f"{fase}Tabla {tabla.iteracion} · {numero}"

    def _dibujar_tabla(self) -> None:
        tabla = self.pasos[self.paso]
        area = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_tabla, self._titulo_tabla(tabla))
        con_razon = bool(tabla.razones)
        encabezados = ["Base", *tabla.columnas, "Solución"] + (["Razón"] if con_razon else [])
        filas = []
        for i, (variable, fila) in enumerate(zip(tabla.base, tabla.filas)):
            razon = ["-" if tabla.razones[i] is None else fraccion(tabla.razones[i])] if con_razon else []
            filas.append([variable, *map(fraccion, fila), *razon])
        filas.append([tabla.etiqueta_objetivo, *map(fraccion, tabla.fila_objetivo), *([""] if con_razon else [])])

        alto_disponible = area.height - 120
        for tamano in (19, 17, 15, 13):
            fuente = self.fuentes.obtener(tamano)
            anchos = [max(fuente.size(t)[0] for t in columna) + 18 for columna in zip(encabezados, *filas)]
            alto_fila = tamano + 9
            if sum(anchos) <= area.width and alto_fila * (len(filas) + 1) <= alto_disponible:
                break
        escala = area.width / sum(anchos)
        anchos = [a * min(escala, 2.0) for a in anchos]
        filas_visibles = max(1, min(len(filas), alto_disponible // alto_fila - 1))
        if filas_visibles < len(filas):
            filas = filas[: filas_visibles - 1] + [filas[-1]]
        xs = [area.x + sum(anchos[:i]) for i in range(len(anchos) + 1)]
        y0 = area.y + 4
        alto_total = alto_fila * (len(filas) + 1)

        k = tabla.columnas.index(tabla.entrante) + 1 if tabla.entrante else None
        r = tabla.base.index(tabla.saliente) if tabla.saliente in tabla.base[: len(filas) - 1] else None
        pygame.draw.rect(self.lienzo, tema.mezclar(tema.PANEL, tema.BORDE, 0.6), (xs[0], y0, xs[-1] - xs[0], alto_fila))
        if k is not None:
            pygame.draw.rect(self.lienzo, tema.mezclar(tema.PANEL, tema.AZUL, 0.22), (xs[k], y0, anchos[k], alto_total))
        if r is not None:
            pygame.draw.rect(self.lienzo, tema.mezclar(tema.PANEL, tema.NARANJA, 0.22),
                             (xs[0], y0 + alto_fila * (r + 1), xs[-1] - xs[0], alto_fila))
        for j, texto in enumerate(encabezados):
            color = tema.AZUL if j == k else tema.TEXTO
            self.fuentes.texto(self.lienzo, texto, (xs[j] + anchos[j] / 2, y0 + 4), tamano, color, True, "centro")
        for i, fila in enumerate(filas):
            y = y0 + alto_fila * (i + 1)
            es_objetivo = i == len(filas) - 1
            if es_objetivo:
                pygame.draw.line(self.lienzo, tema.TEXTO_SUAVE, (xs[0], y), (xs[-1], y), 1)
            for j, texto in enumerate(fila):
                color = tema.NARANJA if i == r and (j == 0 or j == len(fila) - 1 and con_razon) else tema.TEXTO
                self.fuentes.texto(self.lienzo, texto, (xs[j] + anchos[j] / 2, y + 4), tamano, color, es_objetivo or j == 0, "centro")
        for x in xs:
            pygame.draw.line(self.lienzo, tema.BORDE, (x, y0), (x, y0 + alto_total))
        pygame.draw.rect(self.lienzo, tema.BORDE, (xs[0], y0, xs[-1] - xs[0], alto_total), 1)
        if k is not None and r is not None:
            pygame.draw.rect(self.lienzo, tema.AMARILLO, (xs[k], y0 + alto_fila * (r + 1), anchos[k], alto_fila), 2)

        y = y0 + alto_total + 12
        for texto, color in self._explicacion(tabla):
            for linea in self._envolver(texto, 17, area.width):
                self.fuentes.texto(self.lienzo, linea, (area.x + 2, y), 17, color)
                y += 21

    def _explicacion(self, tabla: TablaSimplex) -> list[tuple[str, tuple]]:
        lineas = []
        if self.es_alternativa:
            entra, sale = self.resultado.pivote_alternativo
            lineas.append((f"Entró {entra} (z_j - c_j = 0) y salió {sale}: Z no cambia, es otra solución óptima.", tema.VERDE))
        elif tabla.entrante is None and tabla.fase == 1:
            w = tabla.fila_objetivo[-1]
            if self.resultado.estado == "infactible" and tabla is self.pasos[-1]:
                lineas.append((f"Fin de la fase 1 con W* = {fraccion(w)} < 0: alguna artificial no puede valer 0. PROBLEMA INFACTIBLE.", tema.ROJO))
            else:
                lineas.append((f"Fin de la fase 1 con W* = {fraccion(w)}: se eliminan las artificiales y comienza la fase 2.", tema.VERDE))
        elif tabla.entrante is None:
            texto = "Todos los z_j - c_j son ≥ 0: la tabla es ÓPTIMA."
            if self.resultado.soluciones_multiples:
                texto += " Hay una no básica con z_j - c_j = 0: soluciones múltiples."
            lineas.append((texto, tema.VERDE))
        else:
            k = tabla.columnas.index(tabla.entrante)
            lineas.append((f"Entra {tabla.entrante}: su z_j - c_j = {fraccion(tabla.fila_objetivo[k])} es el más negativo "
                           "(columna azul).", tema.AZUL))
            if tabla.saliente is None:
                lineas.append((f"Ningún coeficiente de la columna {tabla.entrante} es positivo: no hay razón mínima. "
                               "SOLUCIÓN NO ACOTADA.", tema.NARANJA))
            else:
                r = tabla.base.index(tabla.saliente)
                lineas.append((f"Sale {tabla.saliente}: menor razón b_i / a_ik = {fraccion(tabla.razones[r])} (fila naranja); "
                               f"pivote {fraccion(tabla.filas[r][k])} (recuadro amarillo).", tema.NARANJA))
        nombres = self.problema.nombres
        valores = valores_de_tabla(tabla, nombres)
        objetivo = "W" if tabla.fase == 1 else "Z"
        lineas.append(("Solución básica: " + ", ".join(f"{x} = {fraccion(v)}" for x, v in zip(nombres, valores))
                       + f" · {objetivo} = {fraccion(valor_objetivo_de_tabla(tabla))}", tema.TEXTO_SUAVE))
        return lineas

    def _envolver(self, texto: str, tamano: int, ancho: int) -> list[str]:
        fuente = self.fuentes.obtener(tamano)
        lineas, actual = [], ""
        for palabra in texto.split(" "):
            prueba = f"{actual} {palabra}".strip()
            if fuente.size(prueba)[0] <= ancho or not actual:
                actual = prueba
            else:
                lineas.append(actual)
                actual = palabra
        return lineas + [actual]

    def _dibujar_problema(self) -> None:
        area = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_problema, "Problema y forma estándar")
        nombres = self.problema.nombres
        lineas = [(f"{self.problema.sentido} Z = {expresion_lineal(self.problema.objetivo, nombres)}", tema.TEXTO), ("s.a.", tema.TEXTO)]
        for i, r in enumerate(self.problema.restricciones):
            color = COLORES_RESTRICCION[i % len(COLORES_RESTRICCION)] if len(nombres) == 2 else tema.TEXTO
            lineas.append((f"({i + 1}) {expresion_lineal(r.coeficientes, nombres)} {SIGNOS[r.sentido]} {r.lado_derecho:g}", color))
        lineas.append((f"{', '.join(nombres)} ≥ 0", tema.TEXTO))
        agregadas = self.resultado.tablas[0].columnas[len(nombres):]
        grupos = [("Holguras", "s"), ("Excesos", "e"), ("Artificiales", "a")]
        estandar = [f"{titulo}: {', '.join(v for v in agregadas if v.startswith(p))}" for titulo, p in grupos
                    if any(v.startswith(p) for v in agregadas)]
        lineas.append(("", tema.TEXTO))
        lineas += [(texto, tema.TEXTO_SUAVE) for texto in estandar]
        if self.problema.sentido == "min":
            lineas.append(("min Z se resuelve como max (-Z)", tema.TEXTO_SUAVE))
        if self.resultado.usa_dos_fases:
            artificiales = [v for v in agregadas if v.startswith("a")]
            lineas.append((f"Fase 1: max W = -({' + '.join(artificiales)})", tema.TEXTO_SUAVE))
        self._escribir(area, lineas, 16)

    def _escribir(self, area: pygame.Rect, lineas: list[tuple[str, tuple]], tamano: int) -> None:
        y = area.y + 2
        for texto, color in lineas:
            for linea in self._envolver(texto, tamano, area.width - 4):
                if y + tamano > area.bottom:
                    self.fuentes.texto(self.lienzo, "...", (area.x + 2, y - 4), tamano, tema.TEXTO_SUAVE)
                    return
                self.fuentes.texto(self.lienzo, linea, (area.x + 2, y), tamano, color)
                y += tamano + 4

    def _dibujar_resultado(self) -> None:
        area = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_resultado, "Resultado")
        res = self.resultado
        lineas: list[tuple[str, tuple]] = []
        if res.estado == "infactible":
            lineas.append(("PROBLEMA INFACTIBLE", tema.ROJO))
            lineas.append((f"La fase 1 terminó con W* = {fraccion(res.tablas[-1].fila_objetivo[-1])} < 0: ninguna solución "
                           "x ≥ 0 cumple todas las restricciones.", tema.TEXTO))
        elif res.estado == "no_acotado":
            lineas.append(("SOLUCIÓN NO ACOTADA", tema.NARANJA))
            sentido = "+infinito" if self.problema.sentido == "max" else "-infinito"
            lineas.append((f"{res.variable_no_acotada} puede crecer sin límite sin violar ninguna restricción: "
                           f"Z tiende a {sentido}.", tema.TEXTO))
        else:
            lineas.append(("SOLUCIÓN ÓPTIMA" + (" (MÚLTIPLE)" if res.soluciones_multiples else ""), tema.VERDE))
            lineas.append((f"Z* = {res.valor_objetivo:g}", tema.TEXTO))
            lineas.append((", ".join(f"{x} = {v:g}" for x, v in zip(res.variables, res.valores)), tema.TEXTO))
            if res.solucion_alternativa is not None:
                lineas.append(("Otra óptima: " + ", ".join(f"{x} = {v:g}" for x, v in zip(res.variables, res.solucion_alternativa)),
                               tema.VERDE))
            lineas.append(("Precios sombra y_i = ∂Z/∂b_i:", tema.TEXTO_SUAVE))
            for i, (holgura, precio) in enumerate(zip(res.holguras, res.precios_sombra), start=1):
                estado = "activa" if abs(holgura) < 1e-9 else f"holgura {holgura:g}"
                lineas.append((f"({i}) y = {precio:g} · {estado}", tema.TEXTO))
            if res.degenerada:
                lineas.append(("Solución degenerada: los precios sombra pueden no ser únicos.", tema.AMARILLO))
        estado_scipy, _, z_scipy = self.verificacion
        coincide = estado_scipy == res.estado and (z_scipy is None or math.isclose(z_scipy, res.valor_objetivo, abs_tol=1e-7))
        lineas.append(("Verificación con SciPy: " + ("coincide" if coincide else f"NO coincide ({estado_scipy})"),
                       tema.VERDE if coincide else tema.ROJO))
        self._escribir(area, lineas, 17)

    def _puntos_recorrido(self) -> list[tuple[TablaSimplex, tuple[float, ...]]]:
        return [(t, valores_de_tabla(t, self.problema.nombres)) for t in self.pasos[: self.paso + 1]]

    def _dibujar_region(self) -> None:
        panel = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_grafico, "Región factible y recorrido del Simplex")
        x1, x2 = self.problema.nombres
        recorrido = self._puntos_recorrido()
        todos = [v for _, v in ((t, valores_de_tabla(t, self.problema.nombres)) for t in self.pasos)]
        ancho, alto = limites_grafico(self.problema, todos)
        zona = pygame.Rect(panel.x + 44, panel.y + 6, panel.width - 56, 470)

        def punto(x: float, y: float) -> tuple[float, float]:
            return zona.left + x / ancho * zona.width, zona.bottom - y / alto * zona.height

        for i in range(6):
            gx, gy = ancho * i / 5, alto * i / 5
            pygame.draw.line(self.lienzo, tema.REJILLA, punto(gx, 0), punto(gx, alto))
            pygame.draw.line(self.lienzo, tema.REJILLA, punto(0, gy), punto(ancho, gy))
            self.fuentes.texto(self.lienzo, f"{gx:.3g}", (punto(gx, 0)[0], zona.bottom + 3), 14, tema.TEXTO_SUAVE, alineacion="centro")
            self.fuentes.texto(self.lienzo, f"{gy:.3g}", (zona.left - 5, punto(0, gy)[1] - 6), 14, tema.TEXTO_SUAVE, alineacion="derecha")
        self.fuentes.texto(self.lienzo, x1, (zona.right, zona.bottom + 18), 15, tema.TEXTO, alineacion="derecha")
        self.fuentes.texto(self.lienzo, x2, (zona.left + 4, zona.top + 2), 15, tema.TEXTO)

        region = region_factible(self.problema, ancho, alto)
        con_area = len(region) >= 3 and abs(self._area(region)) > 1e-9 * ancho * alto
        if con_area:
            sombra = pygame.Surface(self.lienzo.get_size(), pygame.SRCALPHA)
            pygame.draw.polygon(sombra, (*tema.VERDE, 55), [punto(*p) for p in region])
            self.lienzo.blit(sombra, (0, 0))
            pygame.draw.polygon(self.lienzo, tema.VERDE, [punto(*p) for p in region], 1)

        for i, r in enumerate(self.problema.restricciones):
            tramo = segmento_en_caja(*r.coeficientes, r.lado_derecho, ancho, alto)
            if tramo:
                color = COLORES_RESTRICCION[i % len(COLORES_RESTRICCION)]
                pygame.draw.line(self.lienzo, color, punto(*tramo[0]), punto(*tramo[1]), 2)
        if region and not con_area:
            extremos = sorted(region)
            pygame.draw.line(self.lienzo, tema.VERDE, punto(*extremos[0]), punto(*extremos[-1]), 7)

        tabla_actual, actual = recorrido[-1]
        if tabla_actual.fase == 2 or not self.resultado.usa_dos_fases:
            tramo = segmento_en_caja(*self.problema.objetivo, valor_objetivo_de_tabla(tabla_actual), ancho, alto)
            if tramo:
                tema.linea_discontinua(self.lienzo, tema.TEXTO, punto(*tramo[0]), punto(*tramo[1]), 2)
        if self.resultado.solucion_alternativa is not None and self.paso == len(self.pasos) - 1:
            pygame.draw.line(self.lienzo, tema.VERDE, punto(*self.resultado.valores), punto(*self.resultado.solucion_alternativa), 6)

        puntos = [punto(*v) for _, v in recorrido]
        if len(puntos) >= 2:
            pygame.draw.lines(self.lienzo, tema.AMARILLO, False, puntos, 2)
        for (tabla, _), p in zip(recorrido, puntos):
            pygame.draw.circle(self.lienzo, tema.AMARILLO if tabla.fase == 1 and self.resultado.usa_dos_fases else tema.AZUL, p, 5)
        pygame.draw.circle(self.lienzo, tema.TEXTO, puntos[-1], 9, 2)
        direccion = direccion_no_acotada(tabla_actual, self.problema.nombres)
        if direccion is not None:
            fin = punto(actual[0] + direccion[0], actual[1] + direccion[1])
            dx, dy = fin[0] - puntos[-1][0], fin[1] - puntos[-1][1]
            largo = math.hypot(dx, dy)
            if largo > 1e-9:
                ux, uy = dx / largo, dy / largo
                interior = zona.inflate(-40, -40)
                limites = [140.0]
                if ux:
                    limites.append(((interior.right if ux > 0 else interior.left) - puntos[-1][0]) / ux)
                if uy:
                    limites.append(((interior.bottom if uy > 0 else interior.top) - puntos[-1][1]) / uy)
                avance = max(min(limites), 40.0)
                punta = (puntos[-1][0] + ux * avance, puntos[-1][1] + uy * avance)
                pygame.draw.line(self.lienzo, tema.NARANJA, puntos[-1], punta, 4)
                pygame.draw.polygon(self.lienzo, tema.NARANJA, [
                    (punta[0] + ux * 14, punta[1] + uy * 14),
                    (punta[0] - uy * 8, punta[1] + ux * 8),
                    (punta[0] + uy * 8, punta[1] - ux * 8),
                ])
                a_la_derecha = punta[0] < zona.centerx
                self.fuentes.texto(self.lienzo, "Z crece sin límite", (punta[0] + (14 if a_la_derecha else -14), punta[1] - 8),
                                   17, tema.NARANJA, True, "izquierda" if a_la_derecha else "derecha")
        if not region and self.resultado.estado == "infactible":
            self.fuentes.texto(self.lienzo, "Región factible vacía", (zona.centerx, zona.top + 8), 20, tema.ROJO, True, "centro")

        y = zona.bottom + 40
        for i, r in enumerate(self.problema.restricciones):
            color = COLORES_RESTRICCION[i % len(COLORES_RESTRICCION)]
            pygame.draw.line(self.lienzo, color, (panel.x + 2, y + 8), (panel.x + 22, y + 8), 3)
            self.fuentes.texto(self.lienzo, f"({i + 1}) {expresion_lineal(r.coeficientes, (x1, x2))} {SIGNOS[r.sentido]} {r.lado_derecho:g}",
                               (panel.x + 28, y), 16, color)
            y += 20
        tema.linea_discontinua(self.lienzo, tema.TEXTO, (panel.x + 2, y + 8), (panel.x + 22, y + 8), 2, 5)
        self.fuentes.texto(self.lienzo, "Z constante en la tabla actual", (panel.x + 28, y), 16, tema.TEXTO)
        y += 20
        pygame.draw.circle(self.lienzo, tema.AZUL, (panel.x + 12, y + 8), 5)
        self.fuentes.texto(self.lienzo, "Solución básica de cada tabla (amarillo: fase 1)", (panel.x + 28, y), 16, tema.TEXTO)
        self._dibujar_evolucion(pygame.Rect(panel.x, y + 28, panel.width, panel.bottom - y - 28))

    @staticmethod
    def _area(poligono: list[tuple[float, float]]) -> float:
        return sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poligono, poligono[1:] + poligono[:1])) / 2

    def _dibujar_barras(self) -> None:
        panel = tema.dibujar_panel(self.lienzo, self.fuentes, self.rect_grafico, "Solución básica de la tabla actual")
        nombres = self.problema.nombres
        tabla = self.pasos[self.paso]
        valores = valores_de_tabla(tabla, nombres)
        maximo = max([abs(v) for t in self.pasos for v in valores_de_tabla(t, nombres)] + [1e-9]) * 1.2
        grafico_barras(
            self.lienzo, self.fuentes, pygame.Rect(panel.x, panel.y + 4, panel.width, 230), list(nombres), valores,
            [tema.COLORES_CLUSTER[i % len(tema.COLORES_CLUSTER)] for i in range(len(nombres))], maximo,
            formato=lambda v: f"{v:.4g}",
        )
        y = panel.y + 250
        self.fuentes.texto(self.lienzo, "Uso de cada restricción (a_i·x frente a b_i)", (panel.x + 2, y), 17, tema.TEXTO_SUAVE)
        y += 26
        for i, r in enumerate(self.problema.restricciones[:12], start=1):
            actividad = sum(a * v for a, v in zip(r.coeficientes, valores))
            cumple = {"<=": actividad <= r.lado_derecho + 1e-9, ">=": actividad >= r.lado_derecho - 1e-9,
                      "=": abs(actividad - r.lado_derecho) <= 1e-9}[r.sentido]
            color = tema.VERDE if cumple else tema.ROJO
            pista = pygame.Rect(panel.x + 40, y + 3, 250, 14)
            pygame.draw.rect(self.lienzo, tema.REJILLA, pista, border_radius=3)
            if r.lado_derecho > 0:
                relleno = pista.width * min(max(actividad / r.lado_derecho, 0.0), 1.0)
                pygame.draw.rect(self.lienzo, color, (pista.x, pista.y, relleno, pista.height), border_radius=3)
            self.fuentes.texto(self.lienzo, f"({i})", (panel.x + 2, y + 1), 16, tema.TEXTO)
            self.fuentes.texto(self.lienzo, f"{actividad:.4g} {SIGNOS[r.sentido]} {r.lado_derecho:g}" + ("" if cumple else "  (no cumple)"),
                               (pista.right + 10, y + 1), 16, color)
            y += 24
        self._dibujar_evolucion(pygame.Rect(panel.x, max(y + 10, panel.bottom - 240), panel.width, min(240, panel.bottom - y - 10)))

    def _dibujar_evolucion(self, area: pygame.Rect) -> None:
        """Valor de Z en cada tabla de la fase 2, con un punto en la tabla actual."""
        fase2 = [(i, valor_objetivo_de_tabla(t)) for i, t in enumerate(self.pasos) if t.fase == 2 or not self.resultado.usa_dos_fases]
        if area.height < 80:
            return
        if len(fase2) < 2:
            texto = "Valor de Z: la fase 2 tiene una sola tabla" if fase2 else "Valor de Z: no hay fase 2 (el problema es infactible)"
            self.fuentes.texto(self.lienzo, texto, (area.x + 2, area.y), 15, tema.TEXTO_SUAVE)
            return
        valores = [v for _, v in fase2]
        minimo, maximo = min(valores), max(valores)
        margen = (maximo - minimo) * 0.15 or abs(maximo) * 0.1 or 1.0
        grafico_lineas(
            self.lienzo, self.fuentes, area, [float(i) for i, _ in fase2], [("Z", tema.CIAN, valores)],
            minimo - margen, maximo + margen, formato_y=lambda v: f"{v:.4g}", titulo="Valor de Z en cada tabla de la fase 2",
            x_etiqueta=lambda t: f"paso {t + 1:.0f}",
        )
        actual = next(((i, v) for i, v in fase2 if i == self.paso), None)
        if actual is not None:
            zona = pygame.Rect(area.x + 44, area.y + 16, area.width - 52, area.height - 36)
            x = zona.left + (actual[0] - fase2[0][0]) / (fase2[-1][0] - fase2[0][0]) * zona.width
            y = zona.bottom - (actual[1] - minimo + margen) / (maximo - minimo + 2 * margen) * zona.height
            pygame.draw.circle(self.lienzo, tema.TEXTO, (x, y), 6, 2)

    def _dibujar_mensaje(self) -> None:
        texto, instante = self._mensaje
        if time.time() - instante > 4:
            return
        fuente = self.fuentes.obtener(18, True)
        caja = pygame.Rect(0, 0, fuente.size(texto)[0] + 28, 32)
        caja.midbottom = (self.rect_tabla.centerx, self.rect_tabla.bottom - 8)
        pygame.draw.rect(self.lienzo, (40, 46, 56), caja, border_radius=8)
        pygame.draw.rect(self.lienzo, tema.AZUL, caja, 1, border_radius=8)
        self.fuentes.texto(self.lienzo, texto, (caja.centerx, caja.y + 8), 18, tema.TEXTO, True, "centro")
