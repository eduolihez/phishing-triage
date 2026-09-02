# -*- coding: utf-8 -*-
"""Curvas de triaje: cuanto hay que revisar para cazar los errores.

LA METRICA

Se ordena el correo de mas sospechoso a menos segun una senal de prioridad, y
se recorre la lista preguntando: revisando el X% de los mensajes, ¿que
porcentaje de los errores del ensemble he encontrado?

Una senal buena empuja los errores hacia arriba. Si la curva se pega a la
diagonal, la senal no vale mas que revisar al azar.

Lo que se compara:

    desacuerdo (fuerza de mayoria / desviacion / entropia)
        lo que propone este trabajo

    incertidumbre media
        el baseline honesto: revisar lo que el modelo no tiene claro.
        No hace falta ensemble para esto, asi que el desacuerdo tiene que
        batirlo para justificar su coste.

    azar
        el suelo.

    oraculo
        el techo inalcanzable: conocer los errores de antemano.
"""
import numpy as np


def curva_triaje(prioridad, errores):
    """Devuelve (fraccion_revisada, fraccion_errores_capturados).

    prioridad : mayor = mas sospechoso, se revisa antes
    errores   : booleano, True donde el ensemble se equivoco
    """
    n = len(errores)
    if n == 0 or errores.sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # Desempate aleatorio reproducible: sin esto, los muchos empates de la
    # fuerza de mayoria (solo toma 3 valores) se ordenarian por el orden del
    # fichero, que no es neutral.
    rng = np.random.default_rng(20260902)
    orden = np.lexsort((rng.random(n), -np.asarray(prioridad, dtype=float)))

    capturados = np.cumsum(np.asarray(errores, dtype=float)[orden])
    x = np.arange(1, n + 1) / n
    y = capturados / errores.sum()
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], y])


def area_bajo_curva(x, y):
    """Area normalizada. 0.5 = azar, 1.0 = perfecto."""
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def revisado_para_capturar(x, y, objetivo=0.95):
    """Fraccion que hay que revisar para capturar `objetivo` de los errores.

    Es la cifra que le importa a quien gestiona un SOC: "para no dejar
    escapar el 95%, ¿cuanto trabajo es?".
    """
    idx = np.searchsorted(y, objetivo)
    return float(x[min(idx, len(x) - 1)])


def errores_en_franja(prioridad, errores, franja=0.10):
    """Tasa de error en el decil mas sospechoso frente al menos sospechoso.

    Equivale a las bandas de riesgo del estudio de transcripcion medica: si
    la senal sirve, la franja de arriba concentra los fallos y la de abajo
    esta practicamente limpia.
    """
    n = len(errores)
    k = max(1, int(n * franja))
    orden = np.argsort(-np.asarray(prioridad, dtype=float))
    err = np.asarray(errores, dtype=float)
    return float(err[orden[:k]].mean()), float(err[orden[-k:]].mean())


def resumen(nombre, prioridad, errores, objetivo=0.95):
    x, y = curva_triaje(prioridad, errores)
    alta, baja = errores_en_franja(prioridad, errores)
    return {
        "senal": nombre,
        "auc_triaje": area_bajo_curva(x, y),
        f"revisado_para_{int(objetivo * 100)}": revisado_para_capturar(x, y, objetivo),
        "error_decil_alto": alta,
        "error_decil_bajo": baja,
        "_curva": (x, y),
    }
