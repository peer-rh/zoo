import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from . import GatedRetNet, GatedRetNetConfig
from .data.selective_copying import SelectiveCopyGenerator

BSZ = 64
TRAIN_SEQ_LEN = 1024
EVAL_SEQ_LEN = 1024
EXTRAPOLATE_SEQ_LEN = 4096
N_TO_COPY = 16


def predict(model, X, key, enable_dropout=False):
    state = model.initial_state()
    for i in range(0, X.shape[0], 256):
        y_pred, state = model(X[i : i + 256], state, i, enable_dropout, key)
    return y_pred


@eqx.filter_value_and_grad
def compute_grads(model, data, key):
    X = data[:, :-1]
    y = data[:, -N_TO_COPY:]
    y_pred = jax.vmap(lambda x, k: predict(model, x, k, enable_dropout=True))(
        X, jax.random.split(key, X.shape[0])
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(y_pred[:, -N_TO_COPY:], y)
    return jnp.mean(loss)


@eqx.filter_jit
def step(model, data, optimizer, state, key):
    key, new_key = jax.random.split(key)
    loss, grads = compute_grads(model, data, key)
    updates, new_state = optimizer.update(grads, state, model)

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss, new_key


@eqx.filter_jit
def evaluate(model, data, key):
    X = data[:, :-1]
    y = data[:, -N_TO_COPY:]
    y_pred = jax.vmap(lambda x, k: predict(model, x, k, enable_dropout=False))(
        X, jax.random.split(key, X.shape[0])
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(y_pred[:, -N_TO_COPY:], y)
    accuracy = jnp.mean(jnp.argmax(y_pred[:, -N_TO_COPY:], axis=-1) == y)
    return jnp.mean(loss), accuracy


def train():
    dataGen = SelectiveCopyGenerator(32)
    key = jax.random.PRNGKey(0)
    config = GatedRetNetConfig(n_vocab=32, d_model=64, n_layers=2, n_heads=2, d_ff=128)
    model = GatedRetNet(config, key)
    optimizer = optax.adagrad(1e-4)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_loss = 0.0
    for i in tqdm(range(10000)):
        data_key, step_key, key = jax.random.split(key, 3)
        data = dataGen(BSZ, TRAIN_SEQ_LEN, N_TO_COPY, data_key)
        model, state, loss, key = step(model, data, optimizer, state, step_key)
        train_loss += loss
        if i % 100 == 0:
            eval_key, key = jax.random.split(key)
            eval_data = dataGen(BSZ, EVAL_SEQ_LEN, N_TO_COPY, eval_key)
            eval_loss, eval_acc = evaluate(model, eval_data, eval_key)
            extra_data = dataGen(16, EXTRAPOLATE_SEQ_LEN, N_TO_COPY, eval_key)
            _, extra_acc = evaluate(model, extra_data, eval_key)
            print(
                f"Step: {i} - avg train_loss: {train_loss/100}, eval loss: {eval_loss}, eval acc: {eval_acc}, extra acc: {extra_acc}"
            )
            train_loss = 0.0


if __name__ == "__main__":
    train()
