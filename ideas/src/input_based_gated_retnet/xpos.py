import jax
import jax.numpy as jnp

# NOTE: Currently XPOS mat very well not work for when used with state
# (offset in position )


def apply_xpos(x: jax.Array, inv=False, scale_base=512) -> jax.Array:
    # x: (sqlen, d_model)
    sqlen, dim = x.shape
    assert dim % 2 == 0

    tmp = jnp.arange(0, sqlen)
    theta = 1.0 / (10000 ** (jnp.arange(0, dim / 2) / dim))
    scale = (jnp.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
    scale = (
        scale[None, :] ** ((tmp - (sqlen // 2)) / scale_base)[:, None]
    )  # (sqlen, dim / 2)
    if inv:
        scale = 1.0 / scale
    cos = jnp.cos(tmp[:, None] * theta[None, :]) * scale
    sin = jnp.sin(tmp[:, None] * theta[None, :]) * scale
    out = jnp.zeros_like(x)
    out.at[:, ::2].set(cos * x[:, ::2] - sin * x[:, 1::2])
    out.at[:, 1::2].set(sin * x[:, ::2] + cos * x[:, 1::2])

    return out
