"""Los dos modelos de espanol para PocketSphinx y como se lanza el decodificador.

Los dos son HMM con mezclas gaussianas (estadistica clasica), no redes
neuronales. Se bajan con instalar_modelos.sh a ~/.romeo/modelos/sphinx/.

    ciempiess  UNAM, espanol de Mexico, 581 h de audio, 10 000 senones,
               diccionario de 285 000 palabras.
    cmu        CMU Sphinx 5.2 (VoxForge), espanol general, diccionario y
               modelo de lenguaje de 20 000 palabras.

Se usa el ejecutable `pocketsphinx` de apt (5.0.4), no el modulo de Python:
Ubuntu no lo empaqueta y no se quiere venv. Siempre en modo `live -`, que
lee audio crudo por stdin y corta las frases con su propio detector de voz.

NO usar el modo `single`: medido en la Pi, recarga los modelos en cada
archivo -- 8.9 s por archivo aunque sea silencio. En `live` se cargan una
sola vez. Ademas `live` adapta la normalizacion cepstral (CMN) frase a frase:
la primera frase sale peor y las siguientes mejor.
"""

import os
import subprocess

BASE = os.environ.get("ROMEO_SPHINX_DIR",
                      os.path.expanduser("~/.romeo/modelos/sphinx"))

_CIEMPIESS = os.path.join(BASE, "ciempiess", "CIEMPIESS_Spanish_Models_581h", "Models")
_CMU = os.path.join(BASE, "cmu-es")

MODELOS = {
    "ciempiess": {
        "hmm": os.path.join(_CIEMPIESS, "581HCDCONT10000SPA"),
        "lm": os.path.join(_CIEMPIESS, "581HCDCONT10000SPA.lm.bin"),
        "dict": os.path.join(_CIEMPIESS, "581HCDCONT10000SPA.dic"),
    },
    "cmu": {
        "hmm": os.path.join(_CMU, "cmusphinx-es-5.2", "model_parameters",
                            "voxforge_es_sphinx.cd_ptm_4000"),
        # Descomprimido: pocketsphinx 5 no lee .lm.gz ("Compressed file
        # operation ... not supported") y sigue SIN modelo de lenguaje.
        "lm": os.path.join(_CMU, "es-20k.lm"),
        "dict": os.path.join(_CMU, "es.dict"),
    },
}

SAMPLE_RATE = 16000


def faltantes(nombre):
    return [r for r in MODELOS[nombre].values() if not os.path.exists(r)]


def lanzar(nombre, gramatica=None):
    """Arranca `pocketsphinx live -` y devuelve el Popen (stdin = audio
    crudo s16le 16 kHz mono, stdout = una linea JSON por frase).

    `gramatica` es un archivo JSGF: si se da, en vez del modelo de lenguaje
    solo puede reconocer las frases que ahi se listan (modo comandos)."""
    m = MODELOS[nombre]
    args = ["pocketsphinx", "-hmm", m["hmm"], "-dict", m["dict"],
            "-samprate", str(SAMPLE_RATE)]
    args += ["-jsgf", gramatica] if gramatica else ["-lm", m["lm"]]
    args += ["live", "-"]
    return subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
