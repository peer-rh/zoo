from dataclasses import dataclass
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrng
from equinox import nn

# TODO: Add support for multiple heads
# TODO: Currently this can not use existing state
# - Not very hard to do so for actual state
# , and for the conv, this is also doable (however adds noise to code)


@dataclass
class MambaConfig:
    n_layers: int = 12
    vocab_size: int = 256

    d_model: int = 768
    conv_kernel_size: int = 4
    expansion_factor: int = 4
    d_state: int = 16
    d_delta: int = 48

    @property
    def d_inner(self):
        return self.d_model * self.expansion_factor


class MambaBlock(eqx.Module):
    linMLP: nn.Linear
    conv1: nn.Conv1d
    linOut: nn.Linear

    A: jax.Array
    linBCDelta: nn.Linear
    dt_broadcast: nn.Linear
    D: jax.Array

    config: MambaConfig = eqx.field(static=True)

    def __init__(self, config: MambaConfig, key):
        self.config = config
        mlp_key, out_key, conv_key, bcdelta_key, dt_key = jrng.split(key, 5)
        self.linMLP = nn.Linear(
            config.d_model, config.d_inner * 2, use_bias=False, key=mlp_key
        )
        self.linOut = nn.Linear(
            config.d_inner, config.d_model, use_bias=False, key=out_key
        )
        self.conv1 = nn.Conv1d(
            config.d_inner,
            config.d_inner,
            config.conv_kernel_size,
            use_bias=False,
            groups=config.d_inner,
            padding=[(config.conv_kernel_size - 1, 0)],
            key=conv_key,
        )

        self.A = (jnp.arange(-1, -config.d_state - 1, -1)[None, :]).repeat(
            config.d_inner, 0
        )
        self.D = jnp.ones(config.d_inner)
        self.linBCDelta = nn.Linear(
            config.d_inner,
            config.d_state * 2 + config.d_delta,
            use_bias=False,
            key=bcdelta_key,
        )
        self.dt_broadcast = nn.Linear(
            config.d_delta, config.d_inner, use_bias=True, key=dt_key
        )  # theta is the bias in this layer

    def __call__(self, x):
        """
        x: (sqlen, d_model)
        returns: (sq_len, d_model)
        """

        x, res = jnp.split(
            jax.vmap(self.linMLP)(x), [self.config.d_inner], axis=-1
        )  # (sq_len, d_inner), (sq_len, d_inner)
        x = jax.nn.silu(self.conv1(x.T).T)  # (sq_len, d_inner)

        x = self._ssm(x)

        out = x * jax.nn.silu(res)
        out = jax.vmap(self.linOut)(out)
        return out

    def _ssm(self, x):
        B, C, dt = jnp.split(
            jax.vmap(self.linBCDelta)(x),
            [self.config.d_state, self.config.d_state * 2],
            axis=-1,
        )
        # Note the reason why dt in first compressed is because this is low-rank
        dt = jax.nn.softplus(jax.vmap(self.dt_broadcast)(dt))  # (sqlen, d_inner)

        # Discretisation
        # NOTE: Mamba stores A in log-space, presumably to avoid numerical issues
        # NOTE: Mamba also doesn't use the same discretisation as Zero-Order Hold B, since the
        #       efficiency is increased, while efficacy loss is negligible
        A = jnp.exp(jnp.einsum("id,dn->idn", dt, self.A))  # (sq_len, d_inner, d_state)
        Bx = jnp.einsum("id,in,id->idn", dt, B, x)  # (sq_len, d_inner, d_state)

        def _ssm_step(state, x):
            a, bx, c = x
            state = state * a + bx
            y = jnp.einsum("dn,n->d", state, c)
            return state, y

        _, y = jax.lax.scan(
            _ssm_step, jnp.zeros((self.config.d_inner, self.config.d_state)), (A, Bx, C)
        )

        y += x * self.D
        return y


class Mamba(eqx.Module):
    embed: nn.Embedding
    blocks: List[MambaBlock]
    norms: List[nn.RMSNorm]
    out: nn.Linear

    def __init__(self, config: MambaConfig, key):
        embd_key, block_key, out_key = jrng.split(key, 3)
        self.embed = nn.Embedding(config.vocab_size, config.d_model, key=embd_key)
        self.blocks = [
            MambaBlock(config, key) for key in jrng.split(block_key, config.n_layers)
        ]
        self.norms = [
            nn.RMSNorm(config.d_model, use_bias=False) for _ in range(config.n_layers)
        ]
        self.out = nn.Linear(config.d_model, config.vocab_size, key=out_key)

    def __call__(self, x):
        """
        x: (sqlen,)
        """
        x = jax.vmap(self.embed)(x)
        for i, block in enumerate(self.blocks):
            y = block(jax.vmap(self.norms[i])(x))
            x += y

        return jax.vmap(self.out)(x)
