from . import Config


def getConfig(type: str, n_vocab: int):
    if type == "base":
        return Config(
            n_vocab=n_vocab,
            d_model=256,
            n_layers=6,
            n_heads=4,
            d_mlp=512,
            dropout_ret=0.1,
            dropout_mlp=0.1,
        )
    elif type == "small":
        return Config(
            n_vocab=n_vocab,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_mlp=256,
            dropout_ret=0.1,
            dropout_mlp=0.1,
        )
    elif type == "tiny":
        return Config(
            n_vocab=n_vocab,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_mlp=128,
            dropout_ret=0.1,
            dropout_mlp=0.1,
        )
    else:
        raise ValueError(f"Unknown type: {type}")
