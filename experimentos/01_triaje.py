#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimento principal: ¿sirve el desacuerdo para priorizar la revision?

    python experimentos/01_triaje.py

Entrena las cuatro vistas, mide donde se equivoca el ensemble y compara las
senales de prioridad. Deja las cifras en resultados/ y la grafica que resume
el trabajo.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from triaje import evaluacion as ev  # noqa: E402
from triaje.modelo import (  # noqa: E402
    SEMILLA,
    EnsembleVistas,
    desviacion,
    entropia_voto,
    fuerza_mayoria,
    incertidumbre_media,
)

CORPUS = os.path.join(BASE, "datos", "corpus.jsonl")
RESULTADOS = os.path.join(BASE, "resultados")


def cargar():
    if not os.path.exists(CORPUS):
        print("Falta datos/corpus.jsonl. Ejecuta antes:", file=sys.stderr)
        print("    python scripts/descargar_corpus.py", file=sys.stderr)
        print("    python scripts/construir_dataset.py", file=sys.stderr)
        raise SystemExit(1)
    with open(CORPUS, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    os.makedirs(RESULTADOS, exist_ok=True)
    correos = cargar()

    # El spam cuenta como "no phishing": un buzon real lo recibe, y obliga al
    # modelo a distinguir correo no deseado de correo que roba credenciales,
    # que es la distincion que importa en un SOC.
    y = np.array([1 if c["etiqueta"] == "phishing" else 0 for c in correos])

    print(f"Corpus: {len(correos)} correos ({y.sum()} phishing, {(1 - y).sum()} no phishing)")

    idx_tr, idx_te = train_test_split(
        np.arange(len(correos)), test_size=0.3, random_state=SEMILLA, stratify=y
    )
    tr = [correos[i] for i in idx_tr]
    te = [correos[i] for i in idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    print(f"Entrenamiento: {len(tr)}  |  Prueba: {len(te)}")

    print("\nEntrenando las cuatro vistas...")
    ens = EnsembleVistas().entrenar(tr, y_tr)

    probas = ens.probabilidades(te)
    media = probas.mean(axis=1)
    pred = (media >= 0.5).astype(int)
    errores = pred != y_te

    # ---------------- rendimiento por vista ----------------
    print("\nRendimiento individual (prueba):")
    filas = []
    for i, nombre in enumerate(ens.NOMBRES):
        p = (probas[:, i] >= 0.5).astype(int)
        acc = accuracy_score(y_te, p)
        auc = roc_auc_score(y_te, probas[:, i])
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, p, average="binary", zero_division=0)
        filas.append({"vista": nombre, "exactitud": acc, "auc": auc,
                      "precision": prec, "cobertura": rec, "f1": f1})
        print(f"  {nombre:12s} exactitud={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}")

    acc_ens = accuracy_score(y_te, pred)
    auc_ens = roc_auc_score(y_te, media)
    print(f"  {'ENSEMBLE':12s} exactitud={acc_ens:.4f}  auc={auc_ens:.4f}")
    print(f"\nErrores del ensemble: {int(errores.sum())} de {len(y_te)} ({errores.mean():.2%})")

    if errores.sum() < 10:
        print(
            "\nAVISO: hay tan pocos errores que las curvas de triaje no son\n"
            "fiables. Con una tarea tan facil, el triaje no tiene nada que\n"
            "priorizar. Ver README, apartado de limitaciones.",
            file=sys.stderr,
        )

    # ---------------- comparacion de senales ----------------
    rng = np.random.default_rng(SEMILLA)
    senales = {
        "desacuerdo: fuerza de mayoria": -fuerza_mayoria(probas).astype(float),
        "desacuerdo: desviacion": desviacion(probas),
        "desacuerdo: entropia del voto": entropia_voto(probas),
        "baseline: incertidumbre media": incertidumbre_media(probas),
        "baseline: azar": rng.random(len(y_te)),
        "techo: oraculo": errores.astype(float),
    }

    print("\nSenales de prioridad (ordenadas por AUC de triaje):")
    print(f"  {'senal':32s} {'AUC':>7s} {'rev.95%':>9s} {'err.alto':>9s} {'err.bajo':>9s}")
    resumenes = []
    for nombre, s in senales.items():
        r = ev.resumen(nombre, s, errores)
        resumenes.append(r)
    resumenes.sort(key=lambda r: -r["auc_triaje"])
    for r in resumenes:
        print(f"  {r['senal']:32s} {r['auc_triaje']:7.4f} "
              f"{r['revisado_para_95']:9.2%} {r['error_decil_alto']:9.2%} "
              f"{r['error_decil_bajo']:9.2%}")

    # ---------------- guardar ----------------
    salida = {
        "corpus": {"total": len(correos), "phishing": int(y.sum()),
                   "no_phishing": int((1 - y).sum())},
        "prueba": {"n": len(y_te), "errores_ensemble": int(errores.sum()),
                   "tasa_error": float(errores.mean()),
                   "exactitud_ensemble": float(acc_ens), "auc_ensemble": float(auc_ens)},
        "vistas": filas,
        "senales": [{k: v for k, v in r.items() if not k.startswith("_")} for r in resumenes],
    }
    ruta = os.path.join(RESULTADOS, "triaje.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nCifras -> {os.path.relpath(ruta, BASE)}")

    np.savez_compressed(
        os.path.join(RESULTADOS, "predicciones.npz"),
        probas=probas, y=y_te, errores=errores,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
