# -*- coding: utf-8 -*-
"""El ensemble de cuatro vistas y las medidas de desacuerdo.

La pregunta del proyecto no es "que exactitud alcanza el clasificador", que
es la pregunta de siempre y esta contestada mil veces. Es esta otra:

    cuando el ensemble se equivoca, ¿se le nota?

Si el desacuerdo entre las vistas se concentra donde estan los fallos, un
analista puede revisar una fraccion pequena del correo y aun asi cazar casi
todos los errores. Eso es triaje, y es lo que se mide en evaluacion.py.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import caracteristicas as carac

SEMILLA = 20260902


def _modelo_numerico():
    """Bosque aleatorio para las vistas de rasgos numericos.

    Se elige bosque y no regresion logistica porque estos rasgos interactuan:
    "hay formulario" solo es alarmante junto a "pide contrasena", y un modelo
    lineal no puede expresar eso.
    """
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEMILLA,
        n_jobs=-1,
    )


def _modelo_texto():
    """TF-IDF + regresion logistica sobre el texto visible.

    min_df=3 descarta lo que aparece en menos de tres correos: nombres
    propios, identificadores de un solo mensaje y basura de codificacion, que
    de otro modo permitirian memorizar correos concretos.
    """
    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=3,
            max_df=0.85,
            ngram_range=(1, 2),
            max_features=50000,
            strip_accents="unicode",
        ),
        LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEMILLA
        ),
    )


class EnsembleVistas:
    """Cuatro clasificadores, cada uno con acceso a una parte del correo."""

    NOMBRES = ["remitente", "urls", "estructura", "texto"]

    def __init__(self):
        self.modelos = {}

    def entrenar(self, correos, y):
        for vista in ("remitente", "urls", "estructura"):
            X = carac.matriz(correos, vista)
            modelo = make_pipeline(StandardScaler(), _modelo_numerico())
            modelo.fit(X, y)
            self.modelos[vista] = modelo

        modelo = _modelo_texto()
        modelo.fit(carac.texto_visible(correos), y)
        self.modelos["texto"] = modelo
        return self

    def probabilidades(self, correos):
        """Matriz (n_correos x 4) con P(phishing) segun cada vista."""
        columnas = []
        for vista in self.NOMBRES:
            if vista == "texto":
                X = carac.texto_visible(correos)
            else:
                X = carac.matriz(correos, vista)
            columnas.append(self.modelos[vista].predict_proba(X)[:, 1])
        return np.column_stack(columnas)


# --------------------------------------------------------------------------
#  medidas de desacuerdo
# --------------------------------------------------------------------------
def fuerza_mayoria(probas, umbral=0.5):
    """Cuantas vistas coinciden con el voto mayoritario (entre 2 y 4).

    Es la medida del estudio de transcripcion medica que inspira esto
    ("majority strength"). 4 = unanimidad; 2 = empate a dos, maxima duda.
    """
    votos = (probas >= umbral).astype(int)
    a_favor = votos.sum(axis=1)
    return np.maximum(a_favor, votos.shape[1] - a_favor)


def desviacion(probas):
    """Desviacion tipica de las probabilidades. Usa el margen, no solo el voto.

    Distingue entre cuatro vistas que dicen 0.51 (nadie lo tiene claro) y
    cuatro que dicen 0.99 (todas seguras), cosa que el voto no ve.
    """
    return probas.std(axis=1)


def entropia_voto(probas, umbral=0.5):
    """Entropia binaria del reparto de votos. 0 = unanimidad, 1 = empate."""
    votos = (probas >= umbral).astype(int)
    p = votos.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.nan_to_num(h)


def incertidumbre_media(probas):
    """Cercania a 0.5 de la probabilidad media: el baseline honesto.

    Es lo que haria cualquiera sin ensemble: revisar aquello de lo que el
    modelo no esta seguro. Si el desacuerdo no bate a esto, no aporta nada.
    """
    return 1.0 - 2.0 * np.abs(probas.mean(axis=1) - 0.5)
