import os

import jax

# from jax import api_util
# from jax import lax
from jax import random

# from jax._src import dtypes
# from jax._src import linear_util as lu
# from jax._src import state
from jax._src import test_util as jtu

# from jax._src.pallas import pallas_call
from jax.experimental import pallas as pl

# from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
# import numpy as np


class PallasBaseTest(jtu.JaxTestCase):
    INTERPRET = False

    def setUp(self):
        if not self.INTERPRET:
            if jtu.device_under_test() == "cpu":
                self.skipTest("Only interpret mode supported on CPU")

        super().setUp()

    @classmethod
    def pallas_call(cls, *args, **kwargs):
        return pl.pallas_call(
            *args, interpret=cls.INTERPRET, backend="mosaic_gpu", **kwargs
        )


class OpsTest(PallasBaseTest):
    def test_func(
        self,
        func,
        x_shape_dtype: jax.ShapeDtypeStruct = jax.ShapeDtypeStruct((128,), jnp.float32),
        y_dtype: jnp.dtype = jnp.bool_,
        atol=0,
        rtol=0
    ):
        def kernel(x_ref, y_ref):
            y_ref[...] = func(x_ref[...])

        key = random.key(0)
        x = random.normal(key, x_shape_dtype.shape, dtype=x_shape_dtype.dtype)
        # x.at[0].set(float('nan'))
        # x.at[1].set(float('inf'))
        # x.at[2].set(float('-inf'))

        # x = jnp.array([float('nan'), float('inf'), float('-inf'), 1.0, 0.0, -0.0, -1.5, .3], dtype=x_shape_dtype.dtype)
        assert x.shape == x_shape_dtype.shape

        out = self.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct(x_shape_dtype.shape, y_dtype)
        )(x)
        # tol = 1e-6

        if atol != 0 or rtol != 0:
            self.assertAllClose(out, func(x), atol=atol, rtol=rtol)
        else:
            self.assertArraysEqual(out, func(x))

        print(f"PASSED: {func.__name__}, {x_shape_dtype}, atol={atol}, rtol={rtol}")


if __name__ == "__main__":
    test = OpsTest()
    shape = (256,)

    do_profile = False
    if do_profile:
        out_dir = os.getcwd() + f"/results/mosaic00"
        print(f"Profiling to {out_dir}")
        jax.profiler.start_trace(out_dir)

    # test.test_func(lambda x: 1+x, jax.ShapeDtypeStruct(shape, jnp.float32), jnp.float32)
    # test.test_func(jnp.exp2, jax.ShapeDtypeStruct(shape, jnp.float32), jnp.float32)
    test.test_func(jnp.tanh, jax.ShapeDtypeStruct(shape, jnp.float32), jnp.float32, atol=1e-6)

    if do_profile:
        jax.profiler.stop_trace()

    # test.test_func(jnp.isinf, jax.ShapeDtypeStruct(shape, jnp.float32)) #, jnp.float32)
    # test.test_func(jnp.exp2, jax.ShapeDtypeStruct(shape, jnp.float64), jnp.float64)
    # test.test_func(jnp.exp2, jax.ShapeDtypeStruct(shape, jnp.float16), jnp.float16)
    # test.test_func(jnp.exp2, jax.ShapeDtypeStruct(shape, jnp.bfloat16), jnp.bfloat16)

    # jnp.isfinite = lambda x: jnp.logical_and(~jnp.isnan(x), ~jnp.isinf(x))

    # test.test_func(jnp.sqrt, jax.ShapeDtypeStruct(shape, jnp.float16), jnp.float16)

    # test.test_func(jnp.isfinite, jax.ShapeDtypeStruct(shape, jnp.float32))
    # test.test_func(jnp.isfinite, jax.ShapeDtypeStruct(shape, jnp.float64))
    # test.test_func(jnp.isfinite, jax.ShapeDtypeStruct(shape, jnp.float16))
    # test.test_func(jnp.isfinite, jax.ShapeDtypeStruct(shape, jnp.bfloat16))

    """#test.test_func(jnp.isfinite, jax.ShapeDtypeStruct(shape, jnp.float8_e4m3b11fnuz)) #UnicodeDecodeError: 'utf-8' codec can't decode byte 0xef in position 1186: invalid continuation byte
    #test.test_func(jnp.isfinite, jax.ShapeDtypeStruct(shape, jnp.float8_e4m3fn))   #error: 'llvm.fcmp' op operand #0 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got 'i8'
    #test.test_func(jnp.isfinite, jax.ShapeDtypeStruct(shape, jnp.float8_e4m3fnuz)) #error: 'llvm.fcmp' op operand #0 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got 'i8'
    #test.test_func(jnp.isfinite, jax.ShapeDtypeStruct(shape, jnp.float8_e5m2))     #error: 'llvm.fcmp' op operand #0 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got 'i8'
    #test.test_func(jnp.isfinite, jax.ShapeDtypeStruct(shape, jnp.float8_e5m2fnuz)) #error: 'llvm.fcmp' op operand #0 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got 'i8'

    test.test_func(jnp.isinf, jax.ShapeDtypeStruct(shape, jnp.float32))
    #test.test_func(jnp.isinf, jax.ShapeDtypeStruct(shape, jnp.float64))
    test.test_func(jnp.isinf, jax.ShapeDtypeStruct(shape, jnp.float16))
    test.test_func(jnp.isinf, jax.ShapeDtypeStruct(shape, jnp.bfloat16))

    test.test_func(jnp.isnan, jax.ShapeDtypeStruct(shape, jnp.float32))
    #test.test_func(jnp.isnan, jax.ShapeDtypeStruct(shape, jnp.float64))
    test.test_func(jnp.isnan, jax.ShapeDtypeStruct(shape, jnp.float16))
    test.test_func(jnp.isnan, jax.ShapeDtypeStruct(shape, jnp.bfloat16))"""
