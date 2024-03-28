import jax
import jax.numpy as jnp
from jax import ops


class SelectiveCopyGenerator:
    def __init__(self, n_vocab):
        self.n_vocab = n_vocab

    def __call__(self, bsz, sqlen, n_to_copy, key):
        # TOKENS:
        # - 0: white noise
        # - 1...n_vocab-1: copy from input
        # - n_vocab: Token to tell model to insert COPY Tokens
        key1, key2 = jax.random.split(key, 2)
        tokens = jnp.zeros((bsz, sqlen + 1), jnp.int32)
        tokens = tokens.at[:, -(n_to_copy + 1)].set(self.n_vocab)
        idxs = jax.vmap(
            lambda k: jax.random.choice(
                k, self.n_vocab - 1, (n_to_copy,), replace=False
            ).sort()
        )(jax.random.split(key1, bsz))
        y = jax.random.randint(key2, (bsz, n_to_copy), 1, self.n_vocab)
        tokens = jax.vmap(lambda tok, idx, y: tok.at[idx].set(y))(tokens, idxs, y)
        tokens = tokens.at[:, -n_to_copy:].set(y)
        return tokens
