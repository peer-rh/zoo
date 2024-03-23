import unittest

import jax
import jax.numpy as jnp

from . import GatedRetNet, RetNetConfig


class TestRetNet(unittest.TestCase):
    def test_retnet(self):
        config = RetNetConfig()
        model = GatedRetNet(config, key=jax.random.PRNGKey(0))
        x = jax.random.randint(jax.random.PRNGKey(0), (512,), 0, config.n_vocab)
        out_par, _ = model(x)

        state = [
            jnp.zeros((config.n_heads, config.d_head, config.d_head))
            for _ in range(config.n_layers)
        ]
        out_state = []
        for i in range(0, 512, 128):
            out, state = model(x[i : i + 128], state)
            out_state.append(out)

        out_state = jnp.concatenate(out_state, axis=0)
        self.assertTrue(jnp.allclose(out_par, out_state))


if __name__ == "__main__":
    unittest.main()
