#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convierte los correos crudos en un dataset normalizado (JSONL).

    python scripts/construir_dataset.py

Cada linea del JSONL es un correo con los campos que necesitan las cuatro
vistas del ensemble. La idea es hacer el parseo UNA vez: es lo mas lento y lo
mas propenso a fallos raros de codificacion, y no tiene sentido repetirlo en
cada experimento.

DECISION IMPORTANTE — que se guarda y que no

Se guarda el anyo del mensaje, pero NO para usarlo como caracteristica: se
guarda para poder MEDIR el sesgo temporal en experimentos/02_control_epoca.py.
El correo legitimo del corpus es de 2002 y el phishing de 2004-2007, asi que
cualquier rastro de epoca (versiones de software, formato de los Message-ID,
rutas de los Received) permitiria acertar sin mirar el phishing.

Por eso las cabeceras de transporte se descartan aqui, y solo se conservan
las semanticas: quien dice ser el remitente, adonde contesta y que se ve.
"""
import email
import email.utils
import html
import json
import mailbox
import os
import re
import sys
from email.header import decode_header, make_header

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRUDO = os.path.join(BASE, "datos", "crudo")
EXTRAIDO = os.path.join(BASE, "datos", "extraido")
SALIDA = os.path.join(BASE, "datos", "corpus.jsonl")

MBOX_PHISHING = ["phishing0.mbox", "phishing1.mbox", "phishing2.mbox", "phishing3.mbox"]
CARPETAS = [("easy_ham", "ham"), ("hard_ham", "ham"), ("spam", "spam")]

RE_URL = re.compile(r"""https?://[^\s<>"'\)\]]+""", re.I)
RE_HREF = re.compile(r"""<a\s[^>]*href\s*=\s*["']?([^"'\s>]+)""", re.I)
RE_ETIQUETAS = re.compile(r"<[^>]+>")


def texto_cabecera(msg, nombre):
    """Decodifica una cabecera MIME (=?utf-8?B?...?=) sin reventar."""
    bruto = msg.get(nombre)
    if not bruto:
        return ""
    try:
        return str(make_header(decode_header(bruto))).strip()
    except Exception:
        return str(bruto).strip()


def anyo_de(msg):
    try:
        dt = email.utils.parsedate_to_datetime(msg.get("Date"))
        return dt.year if dt else None
    except Exception:
        return None


def partes(msg):
    """Devuelve (texto_plano, html) concatenando las partes del mensaje."""
    plano, htm = [], []
    if msg.is_multipart():
        for parte in msg.walk():
            if parte.is_multipart():
                continue
            tipo = parte.get_content_type()
            if tipo not in ("text/plain", "text/html"):
                continue
            try:
                carga = parte.get_payload(decode=True)
                if carga is None:
                    continue
                juego = parte.get_content_charset() or "latin-1"
                txt = carga.decode(juego, errors="replace")
            except Exception:
                continue
            (plano if tipo == "text/plain" else htm).append(txt)
    else:
        try:
            carga = msg.get_payload(decode=True)
            txt = (carga or b"").decode(msg.get_content_charset() or "latin-1", "replace")
        except Exception:
            txt = ""
        if msg.get_content_type() == "text/html":
            htm.append(txt)
        else:
            plano.append(txt)
    return "\n".join(plano), "\n".join(htm)


def normalizar(msg, etiqueta, fuente, indice):
    plano, htm = partes(msg)
    # El texto visible es lo que lee la victima: texto plano si lo hay, y si
    # no, el HTML despojado de etiquetas.
    visible = plano if plano.strip() else html.unescape(RE_ETIQUETAS.sub(" ", htm))

    urls = RE_URL.findall(plano) + RE_URL.findall(htm)
    hrefs = [html.unescape(h) for h in RE_HREF.findall(htm)]

    return {
        "id": f"{fuente}:{indice}",
        "etiqueta": etiqueta,
        "fuente": fuente,
        "anyo": anyo_de(msg),
        # --- cabeceras semanticas (no de transporte) ---
        "asunto": texto_cabecera(msg, "Subject"),
        "de": texto_cabecera(msg, "From"),
        "responder_a": texto_cabecera(msg, "Reply-To"),
        "ruta_retorno": texto_cabecera(msg, "Return-Path"),
        "para": texto_cabecera(msg, "To"),
        # --- contenido ---
        "texto": visible[:50000],
        "tiene_html": bool(htm.strip()),
        "html": htm[:50000],
        "urls": urls[:200],
        "hrefs": hrefs[:200],
    }


def main():
    if not os.path.isdir(CRUDO):
        print("Faltan los datos. Ejecuta antes:", file=sys.stderr)
        print("    python scripts/descargar_corpus.py", file=sys.stderr)
        return 1

    registros = []

    print("Leyendo phishing (Nazario)...")
    for nombre in MBOX_PHISHING:
        ruta = os.path.join(CRUDO, nombre)
        if not os.path.exists(ruta):
            print(f"  falta {nombre}, se omite")
            continue
        n = 0
        for i, msg in enumerate(mailbox.mbox(ruta)):
            try:
                registros.append(normalizar(msg, "phishing", nombre, i))
                n += 1
            except Exception:
                pass  # un correo ilegible no debe tumbar el corpus entero
        print(f"  {nombre}: {n}")

    print("Leyendo legitimo y spam (SpamAssassin)...")
    for carpeta, etiqueta in CARPETAS:
        d = os.path.join(EXTRAIDO, carpeta)
        if not os.path.isdir(d):
            print(f"  falta {carpeta}, se omite")
            continue
        n = 0
        for i, fichero in enumerate(sorted(os.listdir(d))):
            ruta = os.path.join(d, fichero)
            if not os.path.isfile(ruta) or fichero.startswith("cmds"):
                continue
            try:
                with open(ruta, "rb") as fh:
                    msg = email.message_from_binary_file(fh)
                registros.append(normalizar(msg, etiqueta, carpeta, i))
                n += 1
            except Exception:
                pass
        print(f"  {carpeta}: {n}")

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(registros)} correos -> {os.path.relpath(SALIDA, BASE)}")
    reparto = {}
    for r in registros:
        reparto[r["etiqueta"]] = reparto.get(r["etiqueta"], 0) + 1
    for k in sorted(reparto):
        print(f"  {k:10s} {reparto[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
