import unittest

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrng

from . import Mamba, MambaConfig


class TestMamba(unittest.TestCase):
    def testMamba(self):
        key = jrng.PRNGKey(0)
        config = MambaConfig()
        mamba = Mamba(config, key)
        X = jrng.randint(key, (64,), 0, config.vocab_size)
        Y = mamba(X)


if __name__ == "__main__":
    unittest.main()
