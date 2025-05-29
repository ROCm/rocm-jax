# path hack must come first
import sys
sys.path = sys.path[1:]

import importlib
import logging
import re
import subprocess

import jax
from jax import numpy as jnp


LOG = logging.getLogger(__name__)


def parse_readelf(output):
    out_text = output.decode("utf8")

    syms = []

    for line in out_text.split("\n")[3:]:
        if not line.strip():
            continue

        _, tag, typ, rest = line.split(" ", 3)
        syms.append((tag, typ.strip("(").strip(")"), rest.strip()))

    return syms


class SharedObject(object):
    def __init__(self, path):
        self.path = path

    def get_rpath(self):
        rpath_tup = next(filter(lambda x: x[1] == "RPATH", self.readelf_dynamic()), None)
        if not rpath_tup:
            return None
        return rpath_tup[2]

    def ldd(self):
        pass

    def dt_needed(self):
        needed = filter(lambda x: x[1] == "NEEDED", self.readelf_dynamic())
        vals = list([n[2] for n in needed])

        libs = []

        pat = re.compile(r"Shared library: \[(.*)\]")
        for v in vals:
            m = pat.match(v)
            if m:
                lib = m.group(1)
                libs.append(lib)

        return libs

    def readelf_dynamic(self):
        out = subprocess.check_output(["readelf", "-d", self.path])
        return parse_readelf(out)



def test_matrix_math():
    print(jax.devices())
    A = jnp.full((100, 100), 4.0, dtype="float32")

    # dot / matmul
    print(jnp.matmul(A, A))

    # element-wise
    print(jnp.multiply(A, A))


def print_link_info():
    #  _hybrid.so  _linalg.so  _prng.so  _rnn.so  _solver.so  _sparse.so  _triton.so
    sos = ["_hybrid", "_linalg", "_prng", "_rnn", "_solver", "_sparse", "_triton"]

    for so in sos:
        mod = None

        try:
            modname = "jax_rocm60_plugin.%s" % so
            print("Loading %r" % modname)
            mod = importlib.import_module(modname)
        except Exception as ex:
            print(ex)

        #mod = getattr(jax_rocm60_plugin, so, None)
        if mod:
            print(mod.__file__)
            obj = SharedObject(mod.__file__)
            print(obj.get_rpath())
            print(obj.dt_needed())

        print()


def main():
    print_link_info()
    test_matrix_math()


if __name__ == "__main__":
    main()
