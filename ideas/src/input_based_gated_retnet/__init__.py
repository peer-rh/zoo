from dataclasses import dataclass
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn

from .xpos import apply_xpos

# Potential Performance Bottlenecks:
# - Einsum and transpositions
# - Application of XPos (How much is recomputed every time)
# - Using Tiled like like FlashAttention


@dataclass
class GatedRetNetConfig:
    n_layers: int = 3
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_vocab: int = 10000
    dropout_prob: float = 0.1

    qkv_bias: bool = False

    @property
    def d_head(self):
        return self.d_model // self.n_heads


class GatedMultiScaleRetention(eqx.Module):
    qkv: nn.Linear
    alpha: nn.Linear
    alpha_base: jnp.ndarray
    config: GatedRetNetConfig = eqx.field(static=True)

    out: nn.Linear
    g_norm: nn.GroupNorm
    gate: nn.Linear

    def __init__(self, config: GatedRetNetConfig, key):
        super().__init__()
        qkv_key, alpha_key, out_key, gate_key = jax.random.split(key, 4)
        self.config = config
        self.qkv = nn.Linear(
            config.d_model, 3 * config.d_model, use_bias=config.qkv_bias, key=qkv_key
        )
        self.alpha = nn.Linear(
            config.d_model, self.config.n_heads, use_bias=True, key=alpha_key
        )
        self.alpha_base = jnp.log(
            jax.random.uniform(
                alpha_key, (self.config.n_heads,), minval=0.95, maxval=0.999
            )
        )
        self.g_norm = nn.GroupNorm(self.config.n_heads, self.config.d_model)
        self.out = nn.Linear(
            config.d_model, config.d_model, use_bias=False, key=out_key
        )
        self.gate = nn.Linear(
            config.d_model, config.d_model, use_bias=False, key=gate_key
        )

    def retention(self, q, k, v, alphas, state):
        sqlen = alphas.shape[1]
        alphas = self.alpha_base[:, None] * 8 * alphas
        k = (1 - jnp.exp(alphas))[:, :, None] * k
        alphas = jnp.cumsum(alphas, axis=1)
        Delta = jnp.tril(jnp.exp(alphas[:, :, None] - alphas[:, None, :]))
        attn = jnp.einsum("hid,hjd,hij->hij", q, k, Delta)
        ret = jnp.einsum("hij,hje->ihe", attn, v)
        new_state = None
        if state is not None:
            new_state = jnp.exp(alphas[:, -1, None, None]) * state
            ret = ret + jnp.einsum("hid, hde, hi->ihe", q, state, jnp.exp(alphas))
            new_state = new_state + jnp.einsum(
                "hid, hie, hi->hde", k, v, Delta[:, -1, :]
            )
        ret = ret.reshape(sqlen, self.config.d_model)
        return ret, new_state

    def __call__(self, x, state=None, offset=0):
        # x: (sqlen, d_model)
        # state: (n_heads, d_head, d_head)
        sqlen = x.shape[0]
        # retention with gated hidden propagation
        q, k, v = (
            jax.vmap(self.qkv)(x)
            .reshape(sqlen, 3, self.config.n_heads, self.config.d_head)
            .transpose((1, 2, 0, 3))
        )
        q = jax.vmap(lambda x: apply_xpos(x, offset))(q)
        k = jax.vmap(lambda x: apply_xpos(x, offset, inv=True))(k)
        alphas = jax.nn.sigmoid(jax.vmap(self.alpha)(x)).T  # (n_heads, sqlen)
        ret, new_state = self.retention(q, k, v, alphas, state)

        # gated hidden propagation
        ret = jax.vmap(self.g_norm)(ret)
        out = jax.vmap(self.out)(jax.nn.swish(jax.vmap(self.gate)(ret)) * ret)

        return out, new_state


class GatedRetNetBlock(eqx.Module):
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm
    drop1: nn.Dropout
    ret: GatedMultiScaleRetention
    lin1: nn.Linear
    lin2: nn.Linear
    drop2: nn.Dropout
    drop3: nn.Dropout

    def __init__(self, config: GatedRetNetConfig, key):
        super().__init__()
        ret_key, lin1_key, lin2_key = jax.random.split(key, 3)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ret = GatedMultiScaleRetention(config, key=ret_key)
        self.lin1 = nn.Linear(config.d_model, config.d_ff, key=lin1_key)
        self.lin2 = nn.Linear(config.d_ff, config.d_model, key=lin2_key)
        self.drop1 = nn.Dropout(config.dropout_prob)
        self.drop2 = nn.Dropout(config.dropout_prob)
        self.drop3 = nn.Dropout(config.dropout_prob)

    def __call__(self, x, state=None, offset=0, enable_dropout=False, key=None):
        key1, key2, key3 = (
            jax.random.split(key, 3) if key is not None else (None, None, None)
        )
        ret, new_state = self.ret(jax.vmap(self.ln1)(x), state, offset)
        ret = self.drop1(ret, inference=not enable_dropout, key=key1)
        x = ret + x

        mlp = jax.nn.gelu(jax.vmap(self.lin1)(x))
        mlp = self.drop2(mlp, inference=not enable_dropout, key=key2)
        mlp = jax.vmap(self.lin2)(mlp)
        mlp = self.drop3(mlp, inference=not enable_dropout, key=key3)

        x = mlp + x
        return x, new_state


class GatedRetNet(eqx.Module):
    embd: nn.Embedding
    out: nn.Linear
    blocks: List[GatedRetNetBlock]
    config: GatedRetNetConfig = eqx.field(static=True)

    def __init__(self, config: GatedRetNetConfig, key):
        super().__init__()
        self.config = config
        embd_key, block_key, out_key = jax.random.split(key, 3)
        self.embd = nn.Embedding(config.n_vocab, config.d_model, key=embd_key)
        self.blocks = [
            GatedRetNetBlock(config, k)
            for k in jax.random.split(block_key, config.n_layers)
        ]
        self.out = nn.Linear(config.d_model, config.n_vocab, key=out_key)

    def __call__(self, x, state=None, offset=0, enable_dropout=False, key=None):
        x = jax.vmap(self.embd)(x)
        new_state = []
        keys = (
            jax.random.split(key, len(self.blocks))
            if key is not None
            else [None] * len(self.blocks)
        )
        if (
            state is None
        ):  # NOTE: Check if this has a performance impact (if it gets jitted away)
            state = [None] * len(self.blocks)

        for s, block, k in zip(state, self.blocks, keys):
            x, ns = block(x, s, offset, enable_dropout, key=k)
            new_state.append(ns)
        x = jax.vmap(self.out)(x)
        return x, new_state

    def initial_state(self):
        return [
            jnp.zeros((self.config.n_heads, self.config.d_head, self.config.d_head))
            for _ in range(self.config.n_layers)
        ]
