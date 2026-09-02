#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera la figura que resume el experimento.

    python experimentos/03_grafica.py

Dos paneles:

  izquierda   curvas de triaje. Cuanto mas arriba, mejor: mas errores
              capturados revisando menos correo.
  derecha     tasa de error por decil de sospecha. Una senal util concentra
              los fallos en los primeros deciles y deja limpios los ultimos.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # sin ventana: esto tiene que correr en CI
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from triaje import evaluacion as ev  # noqa: E402
from triaje.modelo import (  # noqa: E402
    desviacion, entropia_voto, fuerza_mayoria, incertidumbre_media,
)

RESULTADOS = os.path.join(BASE, "resultados")

# Paleta sobria, legible en blanco y negro y distinguible con daltonismo.
COLORES = {
    "desacuerdo: desviacion": "#1f77b4",
    "desacuerdo: fuerza de mayoria": "#4c9be8",
    "baseline: incertidumbre media": "#d62728",
    "baseline: azar": "#999999",
    "techo: oraculo": "#2ca02c",
}


def main():
    ruta = os.path.join(RESULTADOS, "predicciones.npz")
    if not os.path.exists(ruta):
        print("Faltan predicciones. Ejecuta antes:", file=sys.stderr)
        print("    python experimentos/01_triaje.py", file=sys.stderr)
        return 1

    d = np.load(ruta)
    probas, errores = d["probas"], d["errores"].astype(bool)
    rng = np.random.default_rng(20260902)

    senales = {
        "desacuerdo: desviacion": desviacion(probas),
        "desacuerdo: fuerza de mayoria": -fuerza_mayoria(probas).astype(float),
        "baseline: incertidumbre media": incertidumbre_media(probas),
        "baseline: azar": rng.random(len(errores)),
        "techo: oraculo": errores.astype(float),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---------------- panel izquierdo: curvas de triaje ----------------
    for nombre, s in senales.items():
        x, y = ev.curva_triaje(s, errores)
        auc = ev.area_bajo_curva(x, y)
        estilo = "--" if nombre.startswith(("baseline", "techo")) else "-"
        grosor = 1.6 if nombre.startswith(("baseline", "techo")) else 2.4
        ax1.plot(x, y, estilo, lw=grosor, color=COLORES[nombre],
                 label=f"{nombre} (AUC {auc:.3f})")

    ax1.axhline(0.95, color="#666", lw=0.8, ls=":")
    ax1.text(0.015, 0.965, "95 % de los errores", fontsize=8, color="#666")
    ax1.set_xlabel("Fraccion del correo revisada por el analista")
    ax1.set_ylabel("Fraccion de errores del ensemble capturados")
    ax1.set_title("Curvas de triaje", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right", frameon=False)
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)

    # ---------------- panel derecho: error por decil ----------------
    principales = ["desacuerdo: desviacion", "baseline: incertidumbre media"]
    ancho = 0.38
    posiciones = np.arange(10)
    for i, nombre in enumerate(principales):
        s = np.asarray(senales[nombre], dtype=float)
        orden = np.argsort(-s)
        trozos = np.array_split(orden, 10)
        tasas = [errores[t].mean() * 100 for t in trozos]
        ax2.bar(posiciones + (i - 0.5) * ancho, tasas, ancho,
                color=COLORES[nombre], label=nombre)

    ax2.axhline(errores.mean() * 100, color="#333", lw=1, ls="--",
                label=f"tasa global ({errores.mean() * 100:.2f} %)")
    ax2.set_xticks(posiciones)
    ax2.set_xticklabels([f"{i + 1}" for i in posiciones], fontsize=8)
    ax2.set_xlabel("Decil de sospecha (1 = mas sospechoso)")
    ax2.set_ylabel("Errores del ensemble (%)")
    ax2.set_title("Concentracion del error por decil", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.25, axis="y")

    n = len(errores)
    fig.suptitle(
        f"Triaje de phishing por desacuerdo — {n} correos de prueba, "
        f"{int(errores.sum())} errores del ensemble ({errores.mean():.2%})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    salida = os.path.join(RESULTADOS, "triaje.png")
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"Figura -> {os.path.relpath(salida, BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
