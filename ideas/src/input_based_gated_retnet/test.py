import unittest

import jax
import jax.numpy as jnp

from . import GatedMultiScaleRetention, GatedRetNet, GatedRetNetConfig
from .xpos import apply_xpos

# TODO: Add option for dense RetNet as seen in paper


class TestRetNet(unittest.TestCase):
    def test_retnet(self):
        config = GatedRetNetConfig()
        model = GatedRetNet(config, key=jax.random.PRNGKey(0))
        x = jax.random.randint(jax.random.PRNGKey(0), (512,), 0, config.n_vocab)
        out_par, _ = model(x)

        state = [
            jnp.zeros((config.n_heads, config.d_head, config.d_head))
            for _ in range(config.n_layers)
        ]
        out_state = []
        for i in range(0, 512, 32):
            out, state = model(x[i : i + 32], state, i)
            out_state.append(out)

        out_state = jnp.concatenate(out_state, axis=0)
        self.assertTrue(jnp.allclose(out_par, out_state, atol=1e-5))

    def test_retention(self):
        config = GatedRetNetConfig()
        model = GatedMultiScaleRetention(config, key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (512, config.d_model))
        out_par, _ = model(x)

        state = jnp.zeros((config.n_heads, config.d_head, config.d_head))
        out_state = []
        for i in range(0, 512, 32):
            out, state = model(x[i : i + 32], state, i)
            out_state.append(out)

        out_state = jnp.concatenate(out_state, axis=0)
        print(out_state)
        print(out_par)
        self.assertTrue(jnp.allclose(out_par, out_state, atol=1e-5))

    def test_xpos(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (512, 64))
        out = apply_xpos(x)
        out_rec = jnp.concatenate(
            [apply_xpos(x[i : i + 32], i) for i in range(0, 512, 32)]
        )
        self.assertTrue(jnp.allclose(out, out_rec))
        out = apply_xpos(x, inv=True)
        out_rec = jnp.concatenate(
            [apply_xpos(x[i : i + 32], i, inv=True) for i in range(0, 512, 32)]
        )
        self.assertTrue(jnp.allclose(out, out_rec))


if __name__ == "__main__":
    unittest.main()
