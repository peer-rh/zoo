import equinox as eqx
import equinox.nn as nn
import jax


class AttentionBlock(eqx.Module):
    attn: nn.MultiheadAttention
    mlp: nn.MLP

    def __init__(
        self,
        n_heads: int,
        embd_dim: int,
        mlp_dim: int,
        activation=jax.nn.gelu,
        qkv_bias: bool = False,
        *,
        key
    ):
        attn_key, mlp_key = jax.random.split(key, 2)
        self.attn = nn.MultiheadAttention(
            n_heads,
            embd_dim,
            use_query_bias=qkv_bias,
            use_key_bias=qkv_bias,
            use_value_bias=qkv_bias,
            key=attn_key,
        )
        self.mlp = nn.MLP(embd_dim, embd_dim, mlp_dim, 1, act=activation, key=mlp_key)
        self.activation = activation
        self.norm1 = nn.LayerNorm(embd_dim)
        self.norm2 = nn.LayerNorm(embd_dim)

    def __call__(self, x: jax.Array, mask: jax.Array | None):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x
