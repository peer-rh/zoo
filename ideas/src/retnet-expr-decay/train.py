from . import ExpressiveRetNet, Config
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

def train_step(model, x, y, key):

