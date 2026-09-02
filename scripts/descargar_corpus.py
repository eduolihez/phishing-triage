#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Descarga los corpus publicos que usa el experimento.

Los datos NO se versionan en este repositorio: son de terceros y pesan
decenas de MB. Este script los deja en datos/crudo/ de forma reproducible.

    python scripts/descargar_corpus.py

FUENTES

  Phishing  Nazario Phishing Corpus (monkey.org/~jose/phishing/)
            Correos de phishing reales recopilados por Jose Nazario. Se usan
            los mbox 0-3, que cubren 2004-2007.

  Legitimo  SpamAssassin Public Corpus (spamassassin.apache.org)
            easy_ham y hard_ham. hard_ham importa: son correos legitimos que
            "parecen" spam (promociones, HTML recargado), justo los que hacen
            dificil la tarea. Sin ellos el problema es artificialmente facil.

  Spam      SpamAssassin Public Corpus, spam.
            Spam no dirigido, que NO es lo mismo que phishing. Se descarga
            para poder distinguir las dos cosas mas adelante.

NOTA SOBRE TREC-07 Y CEAS-08

  Serian mejores como correo legitimo, porque son de 2007 y coincidirian en
  el tiempo con el phishing de Nazario. Ambos estaban alojados en
  plg.uwaterloo.ca y hoy devuelven 404; ademas su licencia prohibia
  redistribuir cualquier porcion del corpus. Si algun dia vuelven a estar
  disponibles, sustituir el ham de 2002 por el de 2007 eliminaria de raiz el
  sesgo temporal que documenta el README.
"""
import hashlib
import os
import sys
import tarfile
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRUDO = os.path.join(BASE, "datos", "crudo")
EXTRAIDO = os.path.join(BASE, "datos", "extraido")

# Identificarse es lo correcto al descargar de servidores academicos.
UA = "phishing-triage/0.1 (investigacion; +https://github.com/eduolihez/phishing-triage)"

NAZARIO = "https://monkey.org/~jose/phishing/"
SPAMASSASSIN = "https://spamassassin.apache.org/old/publiccorpus/"

DESCARGAS = [
    # (url, nombre local, ¿es tar?)
    (NAZARIO + "phishing0.mbox", "phishing0.mbox", False),
    (NAZARIO + "phishing1.mbox", "phishing1.mbox", False),
    (NAZARIO + "phishing2.mbox", "phishing2.mbox", False),
    (NAZARIO + "phishing3.mbox", "phishing3.mbox", False),
    (SPAMASSASSIN + "20030228_easy_ham.tar.bz2", "easy_ham.tar.bz2", True),
    (SPAMASSASSIN + "20030228_hard_ham.tar.bz2", "hard_ham.tar.bz2", True),
    (SPAMASSASSIN + "20030228_spam.tar.bz2", "spam.tar.bz2", True),
]


def descargar(url, destino):
    if os.path.exists(destino):
        print(f"  ya esta: {os.path.basename(destino)}")
        return
    print(f"  bajando: {os.path.basename(destino)} ... ", end="", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as r, open(destino, "wb") as f:
        f.write(r.read())
    mb = os.path.getsize(destino) / (1024 * 1024)
    print(f"{mb:.1f} MB")


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main():
    os.makedirs(CRUDO, exist_ok=True)
    os.makedirs(EXTRAIDO, exist_ok=True)

    print("Descargando corpus...")
    for url, nombre, _ in DESCARGAS:
        try:
            descargar(url, os.path.join(CRUDO, nombre))
        except Exception as exc:
            print(f"FALLO ({exc})")
            print(
                f"\nNo se pudo descargar {url}\n"
                "Si el servidor esta caido, el experimento no puede continuar.",
                file=sys.stderr,
            )
            return 1

    print("\nExtrayendo los .tar.bz2...")
    for _, nombre, es_tar in DESCARGAS:
        if not es_tar:
            continue
        with tarfile.open(os.path.join(CRUDO, nombre), "r:bz2") as t:
            # filter="data" evita rutas absolutas y enlaces fuera del destino.
            # Sin esto Python 3.14 avisa, y con archivos de terceros conviene.
            try:
                t.extractall(EXTRAIDO, filter="data")
            except TypeError:  # Python < 3.12 no acepta filter
                t.extractall(EXTRAIDO)
        print(f"  {nombre}")

    print("\nHuellas SHA-256 (para poder comprobar que son los mismos datos):")
    for _, nombre, _ in DESCARGAS:
        ruta = os.path.join(CRUDO, nombre)
        print(f"  {sha256(ruta)}  {nombre}")

    print(f"\nListo. Datos en {os.path.relpath(CRUDO, BASE)}")
    print("Siguiente paso: python scripts/construir_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
