import unittest

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrng

from . import Griffin, GriffinConfig


class TestGriffin(unittest.TestCase):
    def testGriffin(self):
        key = jrng.PRNGKey(0)
        config = GriffinConfig()
        griffin = Griffin(config, key)
        X = jrng.randint(key, (64,), 0, config.vocab_size)
        state = griffin._get_initial_state()
        Y = griffin(X)


if __name__ == "__main__":
    unittest.main()
