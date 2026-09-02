# -*- coding: utf-8 -*-
"""Cuatro vistas independientes del mismo correo.

POR QUE CUATRO VISTAS Y NO UN MODELO GRANDE

El experimento mide si el DESACUERDO entre clasificadores sirve para decidir
que revisa una persona. Para que el desacuerdo signifique algo, los errores
de los modelos tienen que ser independientes: si todos miran lo mismo, se
equivocan a la vez y el desacuerdo no informa de nada.

La idea viene de los modelos de verificacion del BSC para catalan, que
entrenan dos copias del mismo modelo con MITADES DISJUNTAS del corpus para
decorrelacionar sus errores. Aqui no hay datos de sobra para partirlos, asi
que se decorrelaciona de otra forma: cada clasificador ve una PARTE DISTINTA
del correo y no puede ver las demas.

    remitente    quien dice ser el que escribe (cabeceras semanticas)
    urls         adonde llevan los enlaces
    texto        que dice el mensaje (TF-IDF del texto visible)
    estructura   como esta construido el HTML

Un phishing con dominio limpio engana a la vista de urls, pero la de
estructura le ve el formulario de contrasena. Ese es el tipo de complemento
que hace util el desacuerdo.

QUE SE DEJA FUERA A PROPOSITO

Nada de fechas, X-Mailer, Received, Message-ID ni versiones de software. El
correo legitimo del corpus es de 2002 y el phishing de 2004-2007: cualquiera
de esos campos permite acertar reconociendo la epoca en vez del phishing.
Ver experimentos/02_control_epoca.py, que mide exactamente cuanta senal
espuria queda pese a estas precauciones.
"""
import math
import re
from urllib.parse import urlparse

import numpy as np

# Marcas suplantadas con mas frecuencia en el corpus de la epoca.
MARCAS = [
    "paypal", "ebay", "citibank", "citi", "barclays", "halifax", "natwest",
    "lloyds", "hsbc", "chase", "wellsfargo", "wamu", "bankofamerica",
    "usbank", "amazon", "visa", "mastercard", "westernunion", "volksbank",
    "postbank", "sparkasse", "abbey", "nationwide", "suntrust", "regions",
]

RE_EMAIL = re.compile(r"[\w\.\-\+]+@([\w\-\.]+)")
RE_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
RE_ETIQUETA_A = re.compile(r"<a\s[^>]*>(.*?)</a>", re.I | re.S)
RE_ETIQUETAS = re.compile(r"<[^>]+>")

URGENCIA = [
    "urgent", "immediately", "verify", "suspend", "suspended", "restricted",
    "confirm", "expire", "expires", "within 24", "click here", "update your",
    "security alert", "unauthorized", "limited", "validate", "reactivate",
]


# --------------------------------------------------------------------------
#  utilidades
# --------------------------------------------------------------------------
def _dominio(direccion):
    """Dominio de una direccion de correo, en minusculas."""
    if not direccion:
        return ""
    m = RE_EMAIL.search(direccion)
    return m.group(1).lower().strip(".") if m else ""


def _dominio_registrable(host):
    """Aproximacion a dominio+TLD sin depender de la lista de sufijos publicos.

    Para 'secure.paypal.com.ru' devuelve 'com.ru', que es justo lo que
    interesa: revela que el dominio real no es paypal.com.
    """
    partes = [p for p in (host or "").lower().split(".") if p]
    return ".".join(partes[-2:]) if len(partes) >= 2 else (partes[0] if partes else "")


def _marca_en(texto):
    t = (texto or "").lower()
    return any(m in t for m in MARCAS)


# --------------------------------------------------------------------------
#  vista 1: remitente
# --------------------------------------------------------------------------
NOMBRES_REMITENTE = [
    "de_vs_responder_a", "de_vs_ruta_retorno", "marca_en_nombre_visible",
    "marca_en_dominio", "marca_visible_sin_marca_dominio", "hay_responder_a",
    "nombre_visible_es_correo", "correo_visible_distinto", "para_vacio",
    "para_es_de", "dominio_de_es_ip", "largo_dominio_de", "asunto_marca",
    "asunto_urgencia",
]


def vista_remitente(c):
    dom_de = _dominio(c.get("de", ""))
    dom_resp = _dominio(c.get("responder_a", ""))
    dom_ret = _dominio(c.get("ruta_retorno", ""))
    de_bruto = (c.get("de", "") or "")
    # Parte visible del From: "Banco Seguro" <x@y.com> -> "Banco Seguro"
    visible = de_bruto.split("<")[0].strip(' "\'')
    asunto = (c.get("asunto", "") or "").lower()

    correos_en_visible = RE_EMAIL.findall(visible)

    return [
        # Contestar a un dominio distinto del que firma es la senal clasica.
        1.0 if (dom_resp and dom_de and dom_resp != dom_de) else 0.0,
        1.0 if (dom_ret and dom_de and dom_ret != dom_de) else 0.0,
        1.0 if _marca_en(visible) else 0.0,
        1.0 if _marca_en(dom_de) else 0.0,
        # Dice ser el banco en el nombre visible pero el dominio no lo es:
        # la suplantacion mas comun y mas barata.
        1.0 if (_marca_en(visible) and not _marca_en(dom_de)) else 0.0,
        1.0 if dom_resp else 0.0,
        1.0 if correos_en_visible else 0.0,
        1.0 if (correos_en_visible and correos_en_visible[0].lower() != dom_de) else 0.0,
        1.0 if not (c.get("para") or "").strip() else 0.0,
        1.0 if (c.get("para") and c.get("de") and c["para"].strip() == c["de"].strip()) else 0.0,
        1.0 if RE_IP.match(dom_de or "") else 0.0,
        float(min(len(dom_de), 60)) / 60.0,
        1.0 if _marca_en(asunto) else 0.0,
        float(sum(1 for p in URGENCIA if p in asunto)) / len(URGENCIA),
    ]


# --------------------------------------------------------------------------
#  vista 2: urls
# --------------------------------------------------------------------------
NOMBRES_URLS = [
    "n_urls", "n_hosts_distintos", "hay_ip_literal", "hay_arroba",
    "max_subdominios", "marca_en_host", "marca_en_ruta_no_host",
    "puerto_no_estandar", "largo_max_url", "hay_codificacion_hex",
    "texto_enlace_distinto", "texto_enlace_es_url", "hay_https",
    "ratio_digitos_host", "hay_guion_en_host",
]


def vista_urls(c):
    urls = list(c.get("urls") or []) + list(c.get("hrefs") or [])
    urls = [u for u in urls if u.lower().startswith(("http://", "https://"))]

    if not urls:
        return [0.0] * len(NOMBRES_URLS)

    hosts, subdominios, largos, digitos = [], [], [], []
    ip_literal = arroba = puerto = hexcod = https = guion = 0
    marca_host = marca_ruta = 0

    for u in urls[:100]:
        try:
            p = urlparse(u)
        except Exception:
            continue
        host = (p.hostname or "").lower()
        hosts.append(host)
        largos.append(len(u))
        subdominios.append(host.count("."))
        if RE_IP.match(host):
            ip_literal = 1
        if "@" in (p.netloc or ""):
            arroba = 1  # http://banco.com@servidor-malo/
        # p.port revienta con puertos codificados en hexadecimal, y el corpus
        # los trae: ':%38%37' es ':87' escrito para despistar a los filtros.
        # Que la URL sea impresentable ya es en si mismo la senal.
        try:
            if p.port and p.port not in (80, 443):
                puerto = 1
        except ValueError:
            puerto = 1
        if "%" in u:
            hexcod = 1
        if p.scheme == "https":
            https = 1
        if "-" in host:
            guion = 1
        if _marca_en(host):
            marca_host = 1
        # La marca aparece en la ruta pero NO en el host: paypal.com.php?x
        if _marca_en(p.path or "") and not _marca_en(host):
            marca_ruta = 1
        if host:
            digitos.append(sum(ch.isdigit() for ch in host) / max(len(host), 1))

    # Texto del enlace distinto del destino real: "www.paypal.com" que va a otro sitio.
    textos = [RE_ETIQUETAS.sub("", t).strip() for t in RE_ETIQUETA_A.findall(c.get("html") or "")]
    hrefs = list(c.get("hrefs") or [])
    discrepancia = 0
    texto_es_url = 0
    for t, h in zip(textos, hrefs):
        if t.lower().startswith(("http://", "https://", "www.")):
            texto_es_url = 1
            # Las dos llamadas van juntas en el try: cualquiera de las dos
            # URLs puede venir deliberadamente malformada.
            try:
                dom_t = _dominio_registrable(urlparse(
                    t if t.startswith("http") else "http://" + t).hostname or "")
                dom_h = _dominio_registrable(urlparse(h).hostname or "")
            except Exception:
                continue
            if dom_t and dom_h and dom_t != dom_h:
                discrepancia = 1

    return [
        math.log1p(len(urls)) / 5.0,
        math.log1p(len(set(hosts))) / 5.0,
        float(ip_literal),
        float(arroba),
        float(min(max(subdominios or [0]), 8)) / 8.0,
        float(marca_host),
        float(marca_ruta),
        float(puerto),
        float(min(max(largos or [0]), 200)) / 200.0,
        float(hexcod),
        float(discrepancia),
        float(texto_es_url),
        float(https),
        float(np.mean(digitos)) if digitos else 0.0,
        float(guion),
    ]


# --------------------------------------------------------------------------
#  vista 4: estructura
# --------------------------------------------------------------------------
NOMBRES_ESTRUCTURA = [
    "tiene_html", "hay_formulario", "hay_campo_password", "hay_campo_oculto",
    "hay_iframe", "hay_script", "n_imagenes", "ratio_texto_html",
    "largo_texto", "hay_tabla", "hay_font_color", "solo_imagen",
    "hay_meta_refresh", "n_etiquetas",
]


def vista_estructura(c):
    h = (c.get("html") or "")
    hl = h.lower()
    texto = (c.get("texto") or "")

    n_img = hl.count("<img")
    n_etiquetas = hl.count("<")
    largo_texto = len(texto.strip())

    return [
        1.0 if c.get("tiene_html") else 0.0,
        1.0 if "<form" in hl else 0.0,
        # Pedir la contrasena dentro del propio correo: casi definitivo.
        1.0 if ('type="password"' in hl or "type=password" in hl) else 0.0,
        1.0 if ('type="hidden"' in hl or "type=hidden" in hl) else 0.0,
        1.0 if "<iframe" in hl else 0.0,
        1.0 if "<script" in hl else 0.0,
        math.log1p(n_img) / 5.0,
        float(largo_texto) / max(len(h), 1) if h else 1.0,
        math.log1p(largo_texto) / 12.0,
        1.0 if "<table" in hl else 0.0,
        1.0 if "<font" in hl else 0.0,
        # Correo que es una sola imagen: evita los filtros de texto.
        1.0 if (n_img > 0 and largo_texto < 200) else 0.0,
        1.0 if "http-equiv" in hl and "refresh" in hl else 0.0,
        math.log1p(n_etiquetas) / 10.0,
    ]


# --------------------------------------------------------------------------
#  API
# --------------------------------------------------------------------------
VISTAS_NUMERICAS = {
    "remitente": (vista_remitente, NOMBRES_REMITENTE),
    "urls": (vista_urls, NOMBRES_URLS),
    "estructura": (vista_estructura, NOMBRES_ESTRUCTURA),
}


def matriz(correos, nombre_vista):
    """Matriz de caracteristicas (n_correos x n_rasgos) para una vista."""
    funcion, _ = VISTAS_NUMERICAS[nombre_vista]
    return np.array([funcion(c) for c in correos], dtype=float)


def texto_visible(correos):
    """Entrada de la vista 'texto': asunto + cuerpo, que es lo que se lee."""
    return [((c.get("asunto") or "") + " \n " + (c.get("texto") or "")) for c in correos]
