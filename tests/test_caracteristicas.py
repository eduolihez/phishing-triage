# -*- coding: utf-8 -*-
"""Pruebas de las vistas y de las metricas de triaje.

Se ejecutan sin el corpus: los correos son sinteticos y las senales de
prioridad, construidas a mano. Asi el CI no tiene que descargar 40 MB de
datos de terceros en cada ejecucion.

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triaje import evaluacion as ev  # noqa: E402
from triaje import caracteristicas as carac  # noqa: E402
from triaje.modelo import (  # noqa: E402
    desviacion, entropia_voto, fuerza_mayoria, incertidumbre_media,
)


def correo(**kw):
    base = {
        "asunto": "", "de": "", "responder_a": "", "ruta_retorno": "",
        "para": "", "texto": "", "html": "", "tiene_html": False,
        "urls": [], "hrefs": [],
    }
    base.update(kw)
    return base


class VistaRemitenteTest(unittest.TestCase):
    def test_reply_to_distinto_se_marca(self):
        c = correo(de="Banco <avisos@banco.com>", responder_a="x@otro-sitio.ru")
        v = carac.vista_remitente(c)
        i = carac.NOMBRES_REMITENTE.index("de_vs_responder_a")
        self.assertEqual(v[i], 1.0)

    def test_reply_to_igual_no_se_marca(self):
        c = correo(de="Banco <avisos@banco.com>", responder_a="otro@banco.com")
        v = carac.vista_remitente(c)
        i = carac.NOMBRES_REMITENTE.index("de_vs_responder_a")
        self.assertEqual(v[i], 0.0)

    def test_marca_en_nombre_pero_no_en_dominio(self):
        """"PayPal" <x@servidor-cualquiera.ru> es la suplantacion tipica."""
        c = correo(de='"PayPal Security" <x@servidor-cualquiera.ru>')
        v = carac.vista_remitente(c)
        i = carac.NOMBRES_REMITENTE.index("marca_visible_sin_marca_dominio")
        self.assertEqual(v[i], 1.0)

    def test_marca_legitima_no_dispara(self):
        c = correo(de='"PayPal" <service@paypal.com>')
        v = carac.vista_remitente(c)
        i = carac.NOMBRES_REMITENTE.index("marca_visible_sin_marca_dominio")
        self.assertEqual(v[i], 0.0)

    def test_longitud_correcta(self):
        self.assertEqual(len(carac.vista_remitente(correo())),
                         len(carac.NOMBRES_REMITENTE))


class VistaUrlsTest(unittest.TestCase):
    def test_ip_literal(self):
        c = correo(urls=["http://192.168.1.1/login.php"])
        v = carac.vista_urls(c)
        self.assertEqual(v[carac.NOMBRES_URLS.index("hay_ip_literal")], 1.0)

    def test_arroba_en_url(self):
        """http://banco.com@malicioso.ru engaña al que lee deprisa."""
        c = correo(urls=["http://banco.com@malicioso.ru/x"])
        v = carac.vista_urls(c)
        self.assertEqual(v[carac.NOMBRES_URLS.index("hay_arroba")], 1.0)

    def test_puerto_codificado_no_revienta(self):
        """Caso real del corpus: ':%38%37' es ':87' escrito para despistar.

        urlparse lanza ValueError al leer .port; si no se captura, el
        experimento entero se cae a mitad.
        """
        c = correo(urls=["http://ejemplo.com:%38%37/login"])
        v = carac.vista_urls(c)  # no debe lanzar
        self.assertEqual(v[carac.NOMBRES_URLS.index("puerto_no_estandar")], 1.0)

    def test_url_malformada_no_revienta(self):
        c = correo(urls=["http://[[[malformada", "http://:::/"],
                   hrefs=["http://???"])
        carac.vista_urls(c)  # basta con que no lance

    def test_sin_urls_da_ceros(self):
        self.assertEqual(carac.vista_urls(correo()), [0.0] * len(carac.NOMBRES_URLS))

    def test_texto_del_enlace_discrepa_del_destino(self):
        c = correo(
            html='<a href="http://malicioso.ru/x">http://www.paypal.com</a>',
            hrefs=["http://malicioso.ru/x"],
            urls=["http://malicioso.ru/x"],
        )
        v = carac.vista_urls(c)
        self.assertEqual(v[carac.NOMBRES_URLS.index("texto_enlace_distinto")], 1.0)


class VistaEstructuraTest(unittest.TestCase):
    def test_campo_password(self):
        c = correo(tiene_html=True,
                   html='<form><input type="password" name="p"></form>')
        v = carac.vista_estructura(c)
        self.assertEqual(v[carac.NOMBRES_ESTRUCTURA.index("hay_campo_password")], 1.0)
        self.assertEqual(v[carac.NOMBRES_ESTRUCTURA.index("hay_formulario")], 1.0)

    def test_solo_imagen(self):
        c = correo(tiene_html=True, html="<img src='x.png'>", texto="hola")
        v = carac.vista_estructura(c)
        self.assertEqual(v[carac.NOMBRES_ESTRUCTURA.index("solo_imagen")], 1.0)

    def test_longitud_correcta(self):
        self.assertEqual(len(carac.vista_estructura(correo())),
                         len(carac.NOMBRES_ESTRUCTURA))


class DesacuerdoTest(unittest.TestCase):
    def test_unanimidad_da_fuerza_maxima(self):
        p = np.array([[0.9, 0.9, 0.9, 0.9]])
        self.assertEqual(fuerza_mayoria(p)[0], 4)
        self.assertAlmostEqual(entropia_voto(p)[0], 0.0)

    def test_empate_da_fuerza_minima(self):
        p = np.array([[0.9, 0.9, 0.1, 0.1]])
        self.assertEqual(fuerza_mayoria(p)[0], 2)
        self.assertAlmostEqual(entropia_voto(p)[0], 1.0)

    def test_incertidumbre_distingue_lo_que_el_voto_no_ve(self):
        """Cuatro vistas diciendo 0.51 y cuatro diciendo 0.99 votan igual.

        La fuerza de mayoria no las distingue; la incertidumbre media si, y
        por eso es un baseline dificil de batir.
        """
        dudoso = np.array([[0.51, 0.52, 0.53, 0.50]])
        seguro = np.array([[0.99, 0.98, 0.97, 0.99]])
        self.assertEqual(fuerza_mayoria(dudoso)[0], fuerza_mayoria(seguro)[0])
        self.assertGreater(incertidumbre_media(dudoso)[0], incertidumbre_media(seguro)[0])
        self.assertGreater(desviacion(np.array([[0.9, 0.1, 0.9, 0.1]]))[0],
                           desviacion(seguro)[0])

    def test_incertidumbre_maxima_en_la_mitad(self):
        self.assertAlmostEqual(incertidumbre_media(np.array([[0.5, 0.5, 0.5, 0.5]]))[0], 1.0)
        self.assertAlmostEqual(incertidumbre_media(np.array([[1.0, 1.0, 1.0, 1.0]]))[0], 0.0)


class TriajeTest(unittest.TestCase):
    def test_oraculo_alcanza_su_techo_teorico(self):
        """El oraculo no llega a AUC 1.0: llega a 1 - tasa_error/2.

        Con 10 errores de 100, la curva sube de 0 a 1 a lo largo del primer
        10% y luego es plana, asi que el area es 0.95. Conviene tenerlo
        presente al leer los resultados: el 0.9908 del experimento real es el
        techo de una tasa de error del 1.83%, no una marca mejorable.
        """
        errores = np.array([True] * 10 + [False] * 90)
        x, y = ev.curva_triaje(errores.astype(float), errores)
        techo = 1.0 - errores.mean() / 2
        self.assertAlmostEqual(ev.area_bajo_curva(x, y), techo, places=2)
        self.assertLessEqual(ev.revisado_para_capturar(x, y, 0.95), 0.11)

    def test_senal_inutil_se_parece_al_azar(self):
        rng = np.random.default_rng(0)
        errores = rng.random(2000) < 0.1
        x, y = ev.curva_triaje(rng.random(2000), errores)
        self.assertAlmostEqual(ev.area_bajo_curva(x, y), 0.5, delta=0.06)

    def test_sin_errores_no_lanza(self):
        errores = np.zeros(50, dtype=bool)
        x, y = ev.curva_triaje(np.arange(50), errores)
        self.assertEqual(len(x), len(y))

    def test_franjas_separan_alto_y_bajo(self):
        errores = np.array([True] * 20 + [False] * 180)
        prioridad = np.linspace(1, 0, 200)  # los primeros son los mas sospechosos
        alto, bajo = ev.errores_en_franja(prioridad, errores)
        self.assertGreater(alto, bajo)
        self.assertEqual(bajo, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
