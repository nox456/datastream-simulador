"""Construcción del reporte PDF con reportlab."""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .analista_ia import AnalisisTexto

AZUL_OSCURO = colors.HexColor("#0d3b66")
GRIS_CLARO = colors.HexColor("#eef2f6")


def _registrar_fuentes() -> tuple[str, str]:
    """DejaVu Sans (incluida con matplotlib) soporta λ, Σ, √, ∂… a diferencia de Helvetica."""
    carpeta = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", str(carpeta / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(carpeta / "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def _f(valor, decimales: int = 2) -> str:
    if valor is None:
        return "-"
    if isinstance(valor, float):
        return f"{valor:.{decimales}f}"
    return str(valor)


class GeneradorReportePDF:
    def __init__(self) -> None:
        normal, negrita = _registrar_fuentes()
        base = getSampleStyleSheet()
        self.estilos = {
            "titulo": ParagraphStyle("titulo", parent=base["Title"], fontName=negrita, fontSize=19, textColor=AZUL_OSCURO),
            "subtitulo": ParagraphStyle("subtitulo", parent=base["Normal"], fontName=normal, fontSize=10, alignment=TA_CENTER, textColor=colors.grey),
            "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=negrita, fontSize=14, textColor=AZUL_OSCURO, spaceBefore=10),
            "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=negrita, fontSize=11.5, textColor=AZUL_OSCURO, spaceBefore=8),
            "normal": ParagraphStyle("normal", parent=base["Normal"], fontName=normal, fontSize=9.5, leading=13.5),
            "vineta": ParagraphStyle("vineta", parent=base["Normal"], fontName=normal, fontSize=9.5, leading=13.5, leftIndent=12, bulletIndent=2),
            "tabla": ParagraphStyle("tabla", parent=base["Normal"], fontName=normal, fontSize=8, leading=10),
            "pie": ParagraphStyle("pie", parent=base["Normal"], fontName=normal, fontSize=8, textColor=colors.grey),
        }
        self.fuente, self.fuente_negrita = normal, negrita

    # ------------------------------------------------------------------ #
    def _p(self, texto: str, estilo: str = "normal") -> Paragraph:
        return Paragraph(texto, self.estilos[estilo])

    def _tabla(self, encabezados: list[str], filas: list[list], anchos: list[float] | None = None) -> Table:
        datos = [[self._p(f"<b>{escape(h)}</b>", "tabla") for h in encabezados]]
        datos += [[self._p(escape(str(c)), "tabla") for c in fila] for fila in filas]
        tabla = Table(datos, colWidths=anchos, repeatRows=1)
        tabla.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), GRIS_CLARO),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d1dc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return tabla

    def _imagen(self, ruta: Path, ancho: float = 17 * cm) -> Image:
        imagen = Image(str(ruta))
        proporcion = imagen.imageHeight / imagen.imageWidth
        imagen.drawWidth, imagen.drawHeight = ancho, ancho * proporcion
        return imagen

    def _texto_ia(self, texto: str) -> list:
        """Convierte el texto con '## títulos', '- viñetas' y **negritas** en párrafos."""
        elementos = []
        for linea in texto.splitlines():
            linea = linea.strip()
            if not linea:
                continue
            # Solo se quita el marcador de título o de viñeta; un "**" inicial es negrita y se conserva.
            texto_linea = re.sub(r"^(#+\s*|[-*•]\s+)", "", linea)
            contenido = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(texto_linea))
            if linea.startswith("#"):
                elementos.append(self._p(re.sub(r"</?b>", "", contenido), "h2"))
            elif re.match(r"^[-*•]\s+", linea):
                elementos.append(Paragraph(contenido, self.estilos["vineta"], bulletText="•"))
            else:
                elementos.append(self._p(contenido))
        return elementos

    # ------------------------------------------------------------------ #
    def generar(
        self,
        resumen: dict,
        analisis: AnalisisTexto,
        graficas: dict[str, Path],
        destino: Path,
        captura: Path | None = None,
    ) -> Path:
        par, ind, colas = resumen["parametros"], resumen["indicadores"], resumen["colas"]
        e = []
        e.append(self._p("Reporte de simulación · DataStream Networks", "titulo"))
        e.append(self._p(
            f"Optimización de la red del centro de datos · generado el {resumen['generado']} · "
            f"{par['tiempo_simulado']:.0f} s simulados · semilla {par['semilla']}", "subtitulo"))
        e.append(Spacer(1, 10))

        # 1. Indicadores
        e.append(self._p("1. Indicadores generales", "h1"))
        filas = [
            ["Tareas generadas", ind["generadas"], "Tareas procesadas", ind["procesadas"]],
            ["Descartadas (espera máx.)", f"{ind['descartadas']} ({ind['porcentaje_descarte']:.1%})", "Retransmitidas", ind["retransmitidas"]],
            ["Interrumpidas por fallas", ind["interrumpidas"], "Fallas de nodos", ind["fallas"]],
            ["Throughput promedio", f"{ind['throughput_promedio']:.2f} tareas/s", "Utilización media", f"{ind['utilizacion_promedio']:.1%}"],
            ["Espera promedio", f"{ind['espera_promedio']:.2f} s", "Tiempo en el sistema", f"{ind['tiempo_sistema_promedio']:.2f} s"],
            ["Cola promedio", f"{ind['cola_promedio']:.1f} tareas", "Cambios de ruta", ind["cambios_ruta"]],
            ["Valor total procesado", f"{ind['valor_procesado']:.0f}", "Tasa de llegada base", f"{par['tasa_llegada_base']:.1f} tareas/s"],
        ]
        e.append(self._tabla(["Indicador", "Valor", "Indicador", "Valor"], filas, [4.6 * cm, 3.9 * cm, 4.6 * cm, 3.9 * cm]))
        tipos = sorted(set(ind["completadas_por_tipo"]) | set(ind["descartadas_por_tipo"]))
        if tipos:
            e.append(Spacer(1, 6))
            e.append(self._tabla(
                ["Tipo de tarea", "Completadas", "Descartadas"],
                [[t, ind["completadas_por_tipo"].get(t, 0), ind["descartadas_por_tipo"].get(t, 0)] for t in tipos],
                [5.5 * cm, 4 * cm, 4 * cm]))
        if captura and captura.is_file():
            e.append(Spacer(1, 8))
            e.append(self._p("Vista del dashboard al momento de generar el reporte:"))
            e.append(self._imagen(captura))
        e.append(self._imagen(graficas["trafico"], 15.5 * cm))

        # Teoría de colas
        e.append(self._p("Referencia de teoría de colas", "h2"))
        if "mmc" in colas:
            mmc = colas["mmc"]
            texto = (
                f"Con λ = {colas['tasa_llegada_observada']:.2f} tareas/s, tiempo medio de servicio "
                f"{colas['tiempo_servicio_medio']:.2f} s y c = {mmc['servidores']} servidores equivalentes "
                f"(capacidad total / CPU media por tarea), un sistema M/M/c tendría ρ = {mmc['utilizacion']:.2f}"
            )
            if mmc["estable"]:
                texto += f", P(espera) = {mmc['prob_espera']:.3f} y W = {mmc['w']:.2f} s."
            else:
                texto += " (inestable: la demanda supera la capacidad)."
            e.append(self._p(texto))
            little = colas["little"]
            e.append(self._p(
                f"Ley de Little: L observado = {little['L_observado']:.1f} tareas frente a λ·W = "
                f"{little['lambda_por_W']:.1f} (diferencia relativa {little['error_relativo']:.1%}). "
                "La simulación agrega efectos que el modelo M/M/c no captura: tránsito por la red, "
                "despacho por lotes, fallas y hora pico."))
            e.append(self._p(
                f"Por eso la espera observada ({ind['espera_promedio']:.2f} s, desde la llegada hasta el inicio del "
                "proceso) es mayor que la del modelo: incluye el viaje del paquete por la red y la espera hasta el "
                "siguiente ciclo de despacho, además de los picos de demanda en los que la cola sí crece."))

        # 2. Módulo 1
        e.append(Spacer(1, 8))
        e.append(self._p("2. Módulo 1 · Programación dinámica", "h1"))
        e.append(self._p("Ruta más corta (recursión hacia adelante por etapas)", "h2"))
        e.append(self._p("f<sub>0</sub>(GW) = 0;  f<sub>k</sub>(y) = min<sub>x</sub> { f<sub>k-1</sub>(x) + d(x, y) }, "
                         "con d(x, y) la latencia actual del enlace (incluye congestión). Tabla de la última recursión:"))
        rutas = resumen["rutas"]
        e.append(self._tabla(
            ["Etapa", "Nodo y", "Candidatos x: f(x)+d(x,y)", "f(y)", "Decisión x*"],
            [[f["etapa"], f["nodo"], f["candidatos"], _f(f["f"]), f["decision"]] for f in rutas["tabla"]],
            [1.4 * cm, 1.8 * cm, 8.6 * cm, 2 * cm, 2.6 * cm]))
        e.append(Spacer(1, 6))
        e.append(self._tabla(
            ["Clúster", "Ruta óptima (DP)", "f* (s)", "Ruta en uso", "Latencia (s)"],
            [[r["cluster"], r["ruta_optima"], _f(r["latencia_optima"]), r["ruta"], _f(r["latencia"])]
             for r in rutas["por_cluster"]],
            [2.4 * cm, 4.9 * cm, 1.6 * cm, 4.9 * cm, 2.2 * cm]))
        e.append(self._p(
            f"Para evitar oscilaciones, la ruta en uso solo se reemplaza si deja de ser válida o si la óptima es al "
            f"menos {rutas['umbral_cambio_ruta']:.0%} más rápida; por eso puede diferir ligeramente de la óptima."))
        motivos = ", ".join(f"{k}: {v}" for k, v in rutas["cambios_por_motivo"].items()) or "sin cambios"
        e.append(self._p(f"Cambios de ruta durante la simulación: {ind['cambios_ruta']} ({motivos})."))

        e.append(self._p("Modelo de volumen-carga (mochila 0/1)", "h2"))
        mochila = resumen["mochila"]
        e.append(self._p(
            "f<sub>i</sub>(w) = max{ f<sub>i-1</sub>(w), v<sub>i</sub> + f<sub>i-1</sub>(w - w<sub>i</sub>) }; "
            f"se resolvió {mochila['ejecuciones']} veces con una utilización media de la capacidad libre del "
            f"{mochila['utilizacion_promedio']:.1%}."))
        destacada = mochila["destacada"]
        if destacada:
            e.append(self._p(
                f"Ejemplo representativo (t = {destacada['t']:.1f} s) en {destacada['servidor']}: capacidad libre "
                f"W = {destacada['capacidad']}, {destacada['total_articulos']} tareas en cola, valor óptimo "
                f"{destacada['valor_total']:.2f} con peso usado {destacada['peso_total']}/{destacada['capacidad']}."))
            e.append(self._tabla(
                ["Tarea", "Peso w_i (CPU)", "Valor v_i", "m_i"],
                [[a["tarea"], a["peso"], _f(a["valor"]), 1 if a["elegida"] else 0] for a in destacada["articulos"]],
                [3.5 * cm, 3 * cm, 3 * cm, 3 * cm]))

        # 3. Módulo 2
        e.append(Spacer(1, 8))
        e.append(self._p("3. Módulo 2 · Optimización no lineal", "h1"))
        term = resumen["termico"]
        e.append(self._p("Análisis térmico (modelo de una variable)", "h2"))
        e.append(self._p(f"T(x) = {escape(term['funcion'])}"))
        e.append(self._p(f"T'(x) = {escape(term['primera_derivada'])}     T''(x) = {escape(term['segunda_derivada'])}"))
        puntos = [[p["tipo"], _f(p["carga"]), _f(p["temperatura"])] for p in term["estacionarios"] + term["inflexiones"]]
        if term["carga_emergencia"] is not None:
            puntos.append(["punto crítico de emergencia", _f(term["carga_emergencia"]), _f(term["temperatura_emergencia"])])
        e.append(self._tabla(["Punto", "Carga x (%)", "T (°C)"], puntos, [7 * cm, 3.5 * cm, 3.5 * cm]))
        e.append(self._imagen(graficas["temperatura"], 15 * cm))

        refri = resumen["refrigeracion"]
        e.append(self._p("Método de gradiente (refrigeración)", "h2"))
        e.append(self._p(
            "E(u) = Σ [A·u<sub>i</sub><super>3</super> + B·L<sub>i</sub><super>2</super>/u<sub>i</sub>] + "
            "C·Σ(u<sub>i</sub> - u<sub>j</sub>)<super>2</super>;  u<super>k+1</super> = Proy(u<super>k</super> - α∇E). "
            f"Consumo medio {refri['energia_promedio']:.2f} kW, {refri['iteraciones_promedio']:.1f} iteraciones por "
            f"ciclo (α = {refri['tasa_aprendizaje']}). Niveles finales: "
            + ", ".join(f"{k} {v:.0%}" for k, v in refri["niveles"].items()) + "."))
        e.append(self._imagen(graficas["refrigeracion"], 15 * cm))

        energia = resumen["energia"]
        e.append(self._p("Presupuesto eléctrico (multiplicadores de Lagrange)", "h2"))
        e.append(self._p(
            "max Σ a<sub>i</sub>√E<sub>i</sub> s.a. ΣE<sub>i</sub> = E<sub>max</sub>  ⇒  "
            "E<sub>i</sub>* = E<sub>max</sub>·a<sub>i</sub><super>2</super>/Σa<sub>j</sub><super>2</super>,  "
            "λ* = √(Σa<sub>i</sub><super>2</super>) / (2√E<sub>max</sub>). "
            f"λ* final = {energia['multiplicador']:.4f}; residuo máximo de ∂L/∂E<sub>i</sub> = "
            f"{energia['verificacion']['max_residuo_dL_dEi']:.1e}."))
        menores = energia["verificacion"]["menores_hessiano_orlado"]
        if menores:
            signos = ", ".join("+" if m > 0 else "−" for m in menores)
            conclusion = "es un máximo" if energia["verificacion"]["es_maximo"] else "no cumple la condición de máximo"
            e.append(self._p(
                f"Condición de segundo orden (Hessiano orlado): los menores principales de orden 3 a "
                f"{len(menores) + 2} tienen signos ({signos}); al alternar empezando por + el punto {conclusion}."))
        e.append(self._tabla(
            ["Clúster", "a_i", "E_i* (kW)", "R_i = a_i√E_i", "Factor de velocidad"],
            [[a["cluster"], _f(a["coeficiente"], 3), _f(a["energia"]), _f(a["rendimiento"], 3), f"×{a['factor_velocidad']:.2f}"]
             for a in energia["asignacion"]],
            [3.2 * cm, 2.5 * cm, 3 * cm, 3.2 * cm, 3.5 * cm]))
        e.append(self._imagen(graficas["lagrange"], 15 * cm))

        # 4. Análisis IA
        e.append(PageBreak())
        e.append(self._p("4. Análisis e interpretación", "h1"))
        e.append(self._p(f"Fuente del análisis: {escape(analisis.fuente)}", "pie"))
        e.extend(self._texto_ia(analisis.texto))

        # 5. Bitácora
        e.append(self._p("5. Bitácora de eventos recientes", "h1"))
        e.append(self._tabla(
            ["t (s)", "Tipo", "Evento"],
            [[f"{ev['t']:.1f}", ev["tipo"], ev["mensaje"]] for ev in resumen["eventos"]],
            [1.8 * cm, 2.4 * cm, 12.6 * cm]))

        destino.parent.mkdir(parents=True, exist_ok=True)
        documento = SimpleDocTemplate(
            str(destino), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
            title="Reporte DataStream Networks", author="Simulador DataStream",
        )
        documento.build(e, onFirstPage=self._pie, onLaterPages=self._pie)
        return destino

    def _pie(self, lienzo, documento) -> None:
        lienzo.saveState()
        lienzo.setFont(self.fuente, 8)
        lienzo.setFillColor(colors.grey)
        lienzo.drawString(2 * cm, 1 * cm, "DataStream Networks · Simulador de optimización (SimPy + Pygame)")
        lienzo.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Página {documento.page}")
        lienzo.restoreState()
