#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control negativo: ¿cuanto del acierto viene de la epoca y no del phishing?

    python experimentos/02_control_epoca.py

EL PROBLEMA

El correo legitimo del corpus es de 2002 y el phishing de 2004-2007. Un
clasificador puede acertar casi siempre reconociendo el ANO -- vocabulario,
productos, formas de escribir de cada momento -- sin aprender nada sobre
phishing. Si eso pasa, el 98% de exactitud del experimento 01 es humo y las
conclusiones sobre triaje no valen.

TRES PRUEBAS

  1. Solo el anyo. Cuanto se acierta usando UNICAMENTE el anyo como rasgo.
     Es el techo del atajo temporal.

  2. Franja solapada. Evaluar solo con phishing de 2002-2004, la epoca del
     ham. Si el rendimiento aguanta, el modelo no vive del calendario; si se
     hunde, si.

  3. Predecir la epoca. Entrenar el mismo modelo de texto para distinguir
     anyo antiguo de moderno DENTRO del phishing. Si le resulta facil, el
     texto lleva marca de epoca y el atajo esta disponible.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from triaje.modelo import SEMILLA, EnsembleVistas, _modelo_texto  # noqa: E402
from triaje.caracteristicas import texto_visible  # noqa: E402

CORPUS = os.path.join(BASE, "datos", "corpus.jsonl")
RESULTADOS = os.path.join(BASE, "resultados")


def cargar():
    with open(CORPUS, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    os.makedirs(RESULTADOS, exist_ok=True)
    correos = cargar()
    y = np.array([1 if c["etiqueta"] == "phishing" else 0 for c in correos])
    anyos = np.array([c.get("anyo") or 0 for c in correos])

    print("Reparto por anyo y clase:")
    for cl, nombre in ((1, "phishing"), (0, "no phishing")):
        cont = Counter(a for a, t in zip(anyos, y) if t == cl and 1995 <= a <= 2026)
        top = ", ".join(f"{a}:{n}" for a, n in sorted(cont.items()))
        print(f"  {nombre:12s} {top}")

    salida = {}

    # ---------------- prueba 1: solo el anyo ----------------
    print("\n[1] Clasificar usando UNICAMENTE el anyo")
    validos = anyos > 0
    Xa = anyos[validos].reshape(-1, 1).astype(float)
    ya = y[validos]
    tr, te = train_test_split(np.arange(len(ya)), test_size=0.3,
                              random_state=SEMILLA, stratify=ya)
    from sklearn.tree import DecisionTreeClassifier

    arbol = DecisionTreeClassifier(max_depth=3, random_state=SEMILLA)
    arbol.fit(Xa[tr], ya[tr])
    p = arbol.predict_proba(Xa[te])[:, 1]
    acc_anyo = accuracy_score(ya[te], (p >= 0.5).astype(int))
    auc_anyo = roc_auc_score(ya[te], p)
    print(f"    exactitud={acc_anyo:.4f}  auc={auc_anyo:.4f}")
    print("    <- este es el atajo disponible: lo que se acierta sin mirar el correo")
    salida["solo_anyo"] = {"exactitud": float(acc_anyo), "auc": float(auc_anyo)}

    # ---------------- prueba 2: franja solapada ----------------
    print("\n[2] Evaluar solo en la franja donde las dos clases coexisten (<=2004)")
    idx_tr, idx_te = train_test_split(np.arange(len(correos)), test_size=0.3,
                                      random_state=SEMILLA, stratify=y)
    ens = EnsembleVistas().entrenar([correos[i] for i in idx_tr], y[idx_tr])

    te_correos = [correos[i] for i in idx_te]
    te_y = y[idx_te]
    te_anyos = anyos[idx_te]

    probas = ens.probabilidades(te_correos)
    media = probas.mean(axis=1)

    mascara = (te_anyos > 0) & (te_anyos <= 2004)
    n_ph = int(te_y[mascara].sum())
    n_no = int((1 - te_y[mascara]).sum())
    print(f"    prueba completa : n={len(te_y)}  exactitud="
          f"{accuracy_score(te_y, (media >= 0.5).astype(int)):.4f}  "
          f"auc={roc_auc_score(te_y, media):.4f}")
    if n_ph >= 10 and n_no >= 10:
        acc_f = accuracy_score(te_y[mascara], (media[mascara] >= 0.5).astype(int))
        auc_f = roc_auc_score(te_y[mascara], media[mascara])
        print(f"    franja <=2004   : n={int(mascara.sum())} "
              f"({n_ph} phishing, {n_no} no)  exactitud={acc_f:.4f}  auc={auc_f:.4f}")
        salida["franja_solapada"] = {"n": int(mascara.sum()), "phishing": n_ph,
                                     "no_phishing": n_no,
                                     "exactitud": float(acc_f), "auc": float(auc_f)}
    else:
        print(f"    franja <=2004: insuficiente ({n_ph} phishing, {n_no} no)")
        salida["franja_solapada"] = None

    # ---------------- prueba 3: predecir la epoca ----------------
    print("\n[3] ¿Lleva el texto marca de epoca? (solo dentro del phishing)")
    ph = [c for c in correos if c["etiqueta"] == "phishing" and (c.get("anyo") or 0) > 0]
    ay = np.array([c["anyo"] for c in ph])
    # 2005 fue el anyo de mayor volumen: parte el corpus en dos mitades utiles.
    etiqueta_epoca = (ay >= 2006).astype(int)
    if etiqueta_epoca.sum() > 50 and (1 - etiqueta_epoca).sum() > 50:
        t_tr, t_te = train_test_split(np.arange(len(ph)), test_size=0.3,
                                      random_state=SEMILLA, stratify=etiqueta_epoca)
        textos = texto_visible(ph)
        m = _modelo_texto()
        m.fit([textos[i] for i in t_tr], etiqueta_epoca[t_tr])
        pe = m.predict_proba([textos[i] for i in t_te])[:, 1]
        acc_e = accuracy_score(etiqueta_epoca[t_te], (pe >= 0.5).astype(int))
        auc_e = roc_auc_score(etiqueta_epoca[t_te], pe)
        print(f"    distinguir 2004-2005 de 2006-2007: exactitud={acc_e:.4f}  auc={auc_e:.4f}")
        print("    <- cuanto mas alto, mas marca de epoca lleva el texto")
        salida["epoca_en_texto"] = {"exactitud": float(acc_e), "auc": float(auc_e)}

    ruta = os.path.join(RESULTADOS, "control_epoca.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nCifras -> {os.path.relpath(ruta, BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
