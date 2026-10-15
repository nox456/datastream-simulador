"""DataStream Networks · Simulador de optimización para redes de centros de datos.

Uso rápido:
    python main.py                       menú interactivo
    python main.py simular               dashboard en tiempo real (Pygame)
    python main.py simular --sin-gui     simulación completa + reporte PDF
    python main.py --help                todos los comandos
"""

import sys

from datastream.cli import main

if __name__ == "__main__":
    sys.exit(main())
