import unittest

import equinox as eqx
import jax
import jax.numpy as jnp

from . import Config, EDMultiheadRetention


class TestEDMultiheadRetention(unittest.TestCase):
    def test(self):
        key = jax.random.PRNGKey(0)
        seq_len = 512
        config = Config(
            n_heads=2,
            d_model=32,
            n_layers=6,
        )
        csz = 128
        model = EDMultiheadRetention(config, key)
        X = jax.random.normal(key, (seq_len, config.d_model))

        def apply_chunk(model, X, key):
            Y_chunk = []
            kvs = model.get_initial_kvs()
            for i in range(0, seq_len, csz):
                Y, kvs = model(X[i : i + csz], kvs, key)
                Y_chunk.append(Y)
            Y_chunk = jnp.concatenate(Y_chunk, axis=0)
            return jnp.mean(Y_chunk), Y_chunk

        def apply_rec(model, X, key):
            Y_rec = []
            kvs = model.get_initial_kvs()
            for i in range(0, seq_len):
                Y, kvs = model.recurrent(X[i], kvs, key)
                Y_rec.append(Y[None, :])
            Y_rec = jnp.concatenate(Y_rec, axis=0)
            return jnp.mean(Y_rec), Y_rec

        (_, Y_chunk), Y_chunk_grad = eqx.filter_value_and_grad(
            apply_chunk, has_aux=True
        )(model, X, key)
        (
            (_, Y_rec),
            Y_rec_grad,
        ) = eqx.filter_value_and_grad(
            apply_rec, has_aux=True
        )(model, X, key)
        print(f"Numerical Stability Forward:")
        print(f"-- Avg: {jnp.abs(Y_chunk - Y_rec).mean()}")
        print(f"-- Max: {jnp.abs(Y_chunk- Y_rec).max()}")
        assert jnp.allclose(Y_chunk, Y_rec, atol=1e-5, rtol=1e-5)

        grad_cmp = cmp_trees(Y_chunk_grad, Y_rec_grad)
        print(f"Numerical Stability Backward:")
        print(f"-- Avg: {grad_cmp[0]}")
        print(f"-- Max: {grad_cmp[1]}")


def cmp_trees(tree1, tree2):
    # Compute the difference between corresponding elements
    diff_tree = jax.tree_util.tree_map(lambda x, y: jnp.abs(x - y), tree1, tree2)

    # Flatten the tree of differences to a list of arrays
    diff_list, _ = jax.tree_util.tree_flatten(diff_tree)

    # Concatenate all arrays into a single array and compute the mean
    total_diff = jnp.concatenate([jnp.ravel(diff) for diff in diff_list])
    mean_diff = jnp.mean(total_diff)
    max_diff = jnp.max(total_diff)

    return mean_diff, max_diff


if __name__ == "__main__":
    unittest.main()
