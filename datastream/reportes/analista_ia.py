"""Análisis de los resultados con IA (API de Claude) y alternativa local.

Si existe la variable de entorno ANTHROPIC_API_KEY (puede definirse en un
archivo .env en la raíz del proyecto) se consulta la API de Claude; si no, o si
la llamada falla, se genera un análisis con reglas locales para que el reporte
PDF siempre pueda producirse.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

MODELO_POR_DEFECTO = "claude-opus-5-5"

INSTRUCCIONES_SISTEMA = (
    "Eres un ingeniero senior de infraestructura cloud especializado en redes de centros de datos, "
    "investigación de operaciones (programación dinámica, optimización no lineal, teoría de colas) "
    "y eficiencia energética. Analizas los resultados de un simulador académico de eventos discretos "
    "de la empresa DataStream Networks. Responde siempre en español, con un tono técnico y claro, "
    "apoyándote en los números que recibes. No inventes datos que no estén en el resumen."
)

FORMATO_RESPUESTA = """Redacta un informe ejecutivo de 450 a 650 palabras con exactamente estas secciones,
cada una iniciada con '## ' seguido del título:
## Diagnóstico general
## Enrutamiento y asignación de servidores (programación dinámica)
## Energía y refrigeración (optimización no lineal)
## Riesgos detectados
## Recomendaciones
Usa párrafos breves y viñetas que empiecen con '- '. No uses tablas."""


@dataclass(frozen=True)
class AnalisisTexto:
    texto: str
    fuente: str  # descripción de quién generó el análisis


def _resumen_para_ia(resumen: dict) -> str:
    datos = {clave: valor for clave, valor in resumen.items() if clave not in {"series", "eventos"}}
    datos["eventos_recientes"] = resumen.get("eventos", [])[-12:]
    return json.dumps(datos, ensure_ascii=False, indent=1, default=str)


class AnalistaClaude:
    """Consulta la API de Claude con el SDK oficial de Anthropic."""

    def __init__(self, modelo: str | None = None):
        import anthropic  # importación diferida: solo se necesita si hay clave

        self._anthropic = anthropic
        self.modelo = modelo or os.environ.get("DATASTREAM_MODELO_IA", MODELO_POR_DEFECTO)
        self.cliente = anthropic.Anthropic(timeout=180.0, max_retries=2)

    def analizar(self, resumen: dict) -> AnalisisTexto:
        respuesta = self.cliente.beta.messages.create(
            model=self.modelo,
            max_tokens=16000,
            # Si los filtros de seguridad rechazan la solicitud, la API reintenta con
            # el modelo de respaldo recomendado en lugar de devolver el rechazo.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium"},
            system=INSTRUCCIONES_SISTEMA,
            messages=[
                {
                    "role": "user",
                    "content": "Resumen de la simulación en formato JSON:\n"
                    f"{_resumen_para_ia(resumen)}\n\n{FORMATO_RESPUESTA}",
                }
            ],
        )
        if respuesta.stop_reason == "refusal":
            raise RuntimeError("la API rechazó generar el análisis")
        texto = "".join(bloque.text for bloque in respuesta.content if bloque.type == "text").strip()
        if not texto:
            raise RuntimeError("la API devolvió una respuesta vacía")
        return AnalisisTexto(texto, f"IA · {respuesta.model}")


class AnalistaLocal:
    """Análisis basado en reglas (se usa cuando la IA no está disponible)."""

    def analizar(self, resumen: dict, motivo: str = "") -> AnalisisTexto:
        ind, rutas, term = resumen["indicadores"], resumen["rutas"], resumen["termico"]
        refri, energia = resumen["refrigeracion"], resumen["energia"]
        lineas = ["## Diagnóstico general"]
        lineas.append(
            f"Se simularon {resumen['parametros']['tiempo_simulado']:.0f} s con {ind['generadas']} tareas generadas, "
            f"{ind['procesadas']} procesadas ({ind['throughput_promedio']:.2f} tareas/s) y una utilización media de "
            f"{ind['utilizacion_promedio']:.0%}. La espera promedio hasta iniciar el proceso fue "
            f"{ind['espera_promedio']:.2f} s y el tiempo total en el sistema {ind['tiempo_sistema_promedio']:.2f} s."
        )
        if ind["porcentaje_descarte"] > 0.05:
            lineas.append(
                f"- El {ind['porcentaje_descarte']:.1%} de las tareas se descartó por superar la espera máxima: "
                "la capacidad es insuficiente en las horas pico."
            )
        else:
            lineas.append(f"- El descarte fue bajo ({ind['porcentaje_descarte']:.1%}); la capacidad cubre la demanda.")

        lineas.append("## Enrutamiento y asignación de servidores (programación dinámica)")
        motivos = ", ".join(f"{k}: {v}" for k, v in rutas["cambios_por_motivo"].items()) or "ninguno"
        lineas.append(
            f"La recursión por etapas se recalculó continuamente y la ruta óptima cambió {ind['cambios_ruta']} veces "
            f"(motivos: {motivos}). Los cambios por congestión reparten el tráfico entre los routers de núcleo."
        )
        lineas.append(
            f"- El modelo de volumen-carga se resolvió {resumen['mochila']['ejecuciones']} veces con una utilización "
            f"media de la capacidad libre del {resumen['mochila']['utilizacion_promedio']:.0%}."
        )
        descartes = ind.get("descartadas_por_tipo", {})
        if descartes:
            peor = max(descartes, key=descartes.get)
            lineas.append(f"- El tipo de tarea con más descartes fue '{peor}' ({descartes[peor]} tareas).")

        lineas.append("## Energía y refrigeración (optimización no lineal)")
        if term["carga_emergencia"] is not None:
            lineas.append(
                f"El análisis de T(x) ubica el punto crítico en {term['carga_emergencia']:.1f}% de carga "
                f"(T ≥ {term['temperatura_emergencia']:.0f} °C); la temperatura máxima observada fue "
                f"{term['temperatura_maxima_observada']:.1f} °C con {term['eventos_emergencia']} activaciones de emergencia."
            )
        lineas.append(
            f"- El método de gradiente mantuvo un consumo de refrigeración medio de {refri['energia_promedio']:.2f} kW "
            f"con {refri['iteraciones_promedio']:.1f} iteraciones por ciclo."
        )
        lineas.append(
            f"- Lagrange repartió {energia['presupuesto']:.0f} kW con un precio sombra medio λ* = "
            f"{energia['multiplicador_promedio']:.4f}: cada kW adicional aportaría ese rendimiento marginal."
        )

        lineas.append("## Riesgos detectados")
        lineas.append(f"- Se registraron {ind['fallas']} fallas de nodos y {ind['retransmitidas']} retransmisiones.")
        if term["eventos_emergencia"]:
            lineas.append("- Hubo servidores que superaron el umbral térmico de emergencia.")

        lineas.append("## Recomendaciones")
        if ind["porcentaje_descarte"] > 0.05:
            lineas.append("- Ampliar la capacidad del clúster más cargado o aumentar el presupuesto eléctrico.")
        lineas.append("- Mantener la redundancia de enlaces entre agregación y acceso para tolerar fallas.")
        lineas.append("- Evitar operar los servidores por encima del punto crítico térmico de forma sostenida.")
        if motivo:
            lineas.append(f"(Análisis generado localmente: {motivo}.)")
        return AnalisisTexto("\n".join(lineas), "reglas locales")


def analizar_resultados(resumen: dict) -> AnalisisTexto:
    """Intenta usar Claude; si no es posible, recurre al análisis local."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return AnalistaLocal().analizar(resumen, "no se encontró ANTHROPIC_API_KEY")
    try:
        import anthropic
    except ImportError:
        return AnalistaLocal().analizar(resumen, "el paquete 'anthropic' no está instalado")
    try:
        return AnalistaClaude().analizar(resumen)
    except anthropic.AuthenticationError:
        motivo = "la API key de Anthropic no es válida"
    except anthropic.RateLimitError:
        motivo = "se alcanzó el límite de solicitudes de la API"
    except anthropic.APIStatusError as error:
        motivo = f"error {error.status_code} de la API"
    except anthropic.APIConnectionError:
        motivo = "no hay conexión con la API"
    except RuntimeError as error:
        motivo = str(error)
    return AnalistaLocal().analizar(resumen, motivo)
