from dataclasses import dataclass
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrng
from equinox import nn


@dataclass
class GriffinConfig:
    layers: List[str] = ["RGLRU", "LocalMQA", "RGLRU"]
    vocab_size: int = 256
    d_model: int = 768

    expansion_factor: int = 4
    d_rnn: int = 1024

    mqa_window: int = 128
    mqa_n_queries: int = 8
    mqa_query_dim: int = 64

    @property
    def mlp_inner(self):
        return self.d_model * self.expansion_factor


class RGLRU(eqx.Module):
    lin1: nn.Linear
    Lambda: jax.Array
    config: GriffinConfig = eqx.field(static=True)

    def __init__(self, config: GriffinConfig, key):
        self.config = config
        lin1_key, lambda_key = jrng.split(key, 2)
        self.lin1 = nn.Linear(config.d_model, config.d_rnn * 2, key=lin1_key)

        # define lambda so, that sigmoid(lambda)**8 in 0.9, 0.9999
        inv_sigmoid = lambda x: jnp.log(1 / (1 - x))
        upper = jnp.power(inv_sigmoid(0.999), 1 / 8)
        lower = jnp.power(inv_sigmoid(0.9), 1 / 8)
        self.Lambda = jrng.uniform(lambda_key, (config.d_model,), lower, upper)

    def __call__(self, x, state):
        # x: (sq_len, d_model)
        # state: (d_rnn,d_model)
        # NOTE: Not really sure if this is correct, paper is not very clear what the op. is and what the sizes are
        #       That being said, this should implement the desired behavior
        inp, rec = jnp.split(
            jax.nn.sigmoid(jax.vmap(self.lin1)(x)), (self.config.d_rnn,), axis=-1
        )
        # c=8 in formula in paper, see Appendix for calculation of a
        a = jnp.exp(
            (-8) * jax.nn.softplus(self.Lambda)[None, :] * rec
        )  # (sq_len, d_rnn)
        gated_x = jnp.einsum(
            "is,is,id->isd", jnp.sqrt(1 - a**2), inp, x
        )  # (sq_len, d_rnn, d_model)

        def rec_step(state, x):
            gated_x, a = x
            h = a[:, None] * state + gated_x
            return h, h

        state, y = jax.lax.scan(rec_step, state, (gated_x, a))
        return y, state


class LocalMQA(eqx.Module):
    qkv: nn.Linear
    config: GriffinConfig = eqx.field(static=True)
    out: nn.Linear

    def __init__(self, config: GriffinConfig, key):
        qkv_key, out_key = jrng.split(key, 2)
        self.qkv = nn.Linear(
            config.d_model,
            config.mqa_query_dim * (config.mqa_n_queries + 2),
            key=qkv_key,
        )
        self.out = nn.Linear(config.d_model, config.d_model, key=out_key)
        self.config = config

    def __call__(self, x):
        # NOTE: This is not efficient for long sequences, removing the real benefit of the Local MQA
        # Unless JAX optimises this in the background
        # For good optim one should probably utilise cuSparse, which isn't well supported in JAX afaik
        sq_len = x.shape[0]
        qkv = jax.vmap(self.qkv)(x)
        k, v, q_s = jnp.split(
            qkv, (self.config.mqa_query_dim, self.config.mqa_query_dim * 2), axis=-1
        )
        q = jnp.reshape(
            q_s, (sq_len, self.config.mqa_n_queries, self.config.mqa_query_dim)
        )
        mask = jnp.tril(jnp.ones((sq_len, sq_len))) - jnp.tril(
            jnp.ones((sq_len, sq_len)), -self.config.mqa_window
        )
        attn = jax.nn.softmax(
            jnp.sqrt(self.config.mqa_query_dim) * jnp.einsum("hid,jd->hij", q, k),
            2,
            where=mask,
            initial=0,
        )
        out = jax.vmap(self.out)(jnp.einsum("hij,jd->hid", attn, v))
        return out


class MLPBlock(eqx.Module):
    lin1: nn.Linear
    lin2: nn.Linear
    mlp_inner: int = eqx.field(static=True)

    def __init__(self, config: GriffinConfig, key):
        key1, key2 = jrng.split(key, 2)
        self.lin1 = nn.Linear(config.d_model, config.mlp_inner, key=key1)
        self.lin2 = nn.Linear(config.mlp_inner, config.d_model, key=key2)
        self.mlp_inner = config.mlp_inner

    def __call__(self, x):
        # x: (d_model)
        a, b = jnp.split(self.lin1(x), [self.mlp_inner], axis=-1)
        x = jax.nn.gelu(a) * b
        return self.lin2(x)


class ResidualBlock(eqx.Module):
    norm1: nn.RMSNorm
    norm2: nn.RMSNorm
    tempmix: LocalMQA | RGLRU
    mlp: MLPBlock
    type: str = eqx.field(static=True)

    def __init__(self, type: str, config: GriffinConfig, key):
        tmp_key, mlp_key = jrng.split(key, 2)
        self.type = type
        if type == "LocalMQA":
            self.tempmix = LocalMQA(config, tmp_key)
        else:
            self.tempmix = RGLRU(config, tmp_key)
        self.mlp = MLPBlock(config, mlp_key)
        self.norm1 = nn.RMSNorm(config.d_model, use_bias=False)
        self.norm2 = nn.RMSNorm(config.d_model, use_bias=False)

    def __call__(self, x, state):
        # x: (sqlen, d_model)
        # This if should be optimised away with jax.jit, so no performance hit
        if type(self.tempmix) == LocalMQA:
            x += self.tempmix(x)
        elif type(self.tempmix) == RGLRU:
            x += self.tempmix(self.norm1(x), state)
        x += jax.vmap(self.mlp)(self.norm2(x))
        return x


class Griffin(eqx.Module):
    embed: nn.Embedding
    blocks: nn.Sequential
    norms: List[nn.RMSNorm]
    out: nn.Linear

    def __init__(self, config: GriffinConfig, key):
        embd_key, block_key, out_key = jrng.split(key, 3)
        self.embed = nn.Embedding(config.vocab_size, config.d_model, key=embd_key)
        self.blocks = nn.Sequential(
            [
                ResidualBlock(type, config, key)
                for key, type in zip(
                    jrng.split(block_key, len(config.layers)), config.layers
                )
            ]
        )
        self.out = nn.Linear(config.d_model, config.vocab_size, key=out_key)

    def __call__(self, x):
        """
        x: (sqlen,)
        """
        x = jax.vmap(self.embed)(x)
        x = self.blocks(x)
        return jax.vmap(self.out)(x)
