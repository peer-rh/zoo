import jax
import jax.numpy as jnp


def apply_rope(x: jax.Array, theta: jax.Array) -> jax.Array:
    # x: (sqlen, d_model)
    # theta: (d_model/2)
    tmp = jnp.arange(0, x.shape[0])
    cos = jnp.cos(tmp[:, None] * theta[None,:])
    sin = jnp.sin(tmp[:, None] * theta[None,:])
    out = jnp.zeros_like(x)
    out.at[:, ::2].set(cos * x[:, ::2] - sin * x[:, 1::2])
    out.at[:, 1::2].set(sin * x[:, ::2] + cos * x[:, 1::2])
    return out
