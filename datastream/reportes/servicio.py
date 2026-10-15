"""Caso de uso: generar el reporte PDF (gráficas + análisis IA + documento)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..configuracion import RAIZ_PROYECTO
from .analista_ia import analizar_resultados
from .generador_pdf import GeneradorReportePDF
from .graficas import generar_graficas

DIRECTORIO_REPORTES = RAIZ_PROYECTO / "reportes"


@dataclass(frozen=True)
class ResultadoReporte:
    ruta: Path
    fuente_analisis: str


def generar_reporte(resumen: dict, captura: Path | None = None, directorio: Path = DIRECTORIO_REPORTES) -> ResultadoReporte:
    analisis = analizar_resultados(resumen)
    destino = directorio / f"reporte_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    with tempfile.TemporaryDirectory() as temporal:
        graficas = generar_graficas(resumen, Path(temporal))
        GeneradorReportePDF().generar(resumen, analisis, graficas, destino, captura)
    return ResultadoReporte(destino, analisis.fuente)
