from dataclasses import dataclass
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn

# TODO: Add dropout to the model
# TODO: Constrain alpha to be in range where numerical stability is guaranteed and < 1


@dataclass
class Config:
    n_heads: int
    d_model: int
    d_mlp: int
    n_layers: int
    n_vocab: int = 256

    dropout_ret: float = 0.1
    dropout_mlp: float = 0.1


class EDMultiheadRetention(eqx.Module):
    """
    More expressive-decay multihead retention (This is the unique part of the model)
    """

    qkv: nn.Linear
    alpha: jax.Array
    config: Config = eqx.field(static=True)

    def __init__(self, config: Config, key):
        head_dim = config.d_model // config.n_heads
        self.config = config
        qkv_key, alpha_key = jax.random.split(key, 2)
        self.qkv = nn.Linear(
            config.d_model, 3 * config.d_model, use_bias=False, key=qkv_key
        )
        # NOTE: These values may be suboptimal
        self.alpha = (
            jax.random.truncated_normal(alpha_key, -1, 1, (config.n_heads, head_dim))
            / 5
            + 0.8
        )

    def get_initial_kvs(self):
        return jnp.zeros(
            (
                self.config.n_heads,
                self.config.d_model // self.config.n_heads,
                self.config.d_model // self.config.n_heads,
            )
        )

    def __call__(self, x: jax.Array, kvs: jax.Array):
        """
        x: (seq, d_model)
        kvs: (n_heads, d_model // n_heads)
        NOTE: seq is restricted by the numerical stability (estimate values)
        Usually 128 should work
        """
        sqlen, _ = x.shape

        qkv = jax.vmap(self.qkv)(x)
        q, k, v = qkv.reshape(
            sqlen, 3, self.config.n_heads, self.config.d_model // self.config.n_heads
        ).transpose(1, 2, 0, 3)

        q *= self.alpha[:, None, :] ** jnp.arange(-sqlen + 1, 1, 1)[None, :, None]
        q /= jnp.sqrt(self.config.d_model)
        k *= self.alpha[:, None, :] ** jnp.arange(sqlen - 1, -1, -1)[None, :, None]
        k /= jnp.sqrt(self.config.d_model)

        attn = jnp.tril(jnp.einsum("hid,hjd->hij", q, k))  # (n_heads, seq, seq)
        out = jax.nn.gelu(attn @ v + q @ (kvs * self.alpha[:, :, None] ** sqlen))
        out = out.transpose(1, 0, 2).reshape(sqlen, self.config.d_model)

        new_kvs = kvs * (self.alpha[:, :, None] ** (sqlen))
        new_kvs += jnp.einsum("hdi,hdj->hij", k, v)

        return out, new_kvs

    def recurrent(self, x: jax.Array, kvs: jax.Array):
        """
        NOTE: It is better to use call with seqlen 1, since this is used to ensure correctness in test_case
        x: (d_model)
        kvs: (n_heads, head_dim, head_dim)
        """
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(
            3, self.config.n_heads, self.config.d_model // self.config.n_heads
        )
        q /= jnp.sqrt(self.config.d_model)
        k /= jnp.sqrt(self.config.d_model)

        new_kvs = jnp.einsum("hi,hj->hij", k, v) + kvs * self.alpha[:, :, None]

        out = jax.nn.gelu(q[:, None, :] @ new_kvs).reshape(self.config.d_model)
        return out, new_kvs


class GatedBlock(eqx.Module):
    retention: EDMultiheadRetention
    g: nn.Linear
    mlp: nn.MLP
    drop1: nn.Dropout
    ln1: nn.LayerNorm
    drop2: nn.Dropout
    ln2: nn.LayerNorm

    def __init__(self, config: Config, key):
        retention_key, mlp_key, ln1_key, ln2_key = jax.random.split(key, 4)
        self.retention = EDMultiheadRetention(config, retention_key)
        self.g = nn.Linear(config.d_model, config.d_model, use_bias=False, key=mlp_key)
        self.mlp = nn.MLP(
            config.d_model,
            config.d_model,
            config.d_mlp,
            2,
            activation=jax.nn.gelu,
            key=mlp_key,
        )
        self.ln1 = nn.LayerNorm(
            config.d_model,
            use_bias=False,
        )
        self.ln2 = nn.LayerNorm(
            config.d_model,
            use_bias=False,
        )
        self.drop1 = nn.Dropout(config.dropout_ret)
        self.drop2 = nn.Dropout(config.dropout_mlp)

    def __call__(self, x, kvs, key):
        key1, key2 = jax.random.split(key, 2)
        y, kvs = self.retention(jax.vmap(self.ln1)(x), kvs)
        y = jax.nn.swish(jax.vmap(self.g)(y))  # Gate inspired by RetNet
        x += self.drop1(y, key=key1)
        y = jax.vmap(self.mlp)(jax.vmap(self.ln2)(x))
        x += self.drop2(y, key=key2)
        return x, kvs


class ExpressiveRetNet(eqx.Module):
    config: Config = eqx.field(static=True)
    blocks: List[GatedBlock]
    emb: nn.Embedding
    out: nn.Linear

    def __init__(self, config: Config, key):
        emb_key, out_key, block_key = jax.random.split(key, 3)
        self.config = config
        self.emb = nn.Embedding(config.n_vocab, config.d_model, key=emb_key)
        self.out = nn.Linear(config.d_model, config.n_vocab, key=out_key)
        self.blocks = [
            GatedBlock(config, key)
            for key in jax.random.split(block_key, config.n_layers)
        ]

    def _initial_kvs(self):
        return jnp.stack([block.retention.get_initial_kvs() for block in self.blocks])

    def __call__(self, x, kvs, key):
        x = jax.vmap(self.emb)(x)
        new_kvs = []
        for i, (block, key) in enumerate(
            zip(self.blocks, jax.random.split(key, len(self.blocks)))
        ):
            x, new_kv = block(x, kvs[i], key)
            new_kvs.append(new_kv)
        new_kvs = jnp.stack(new_kvs)
        return jax.vmap(self.out)(x), new_kvs
