from src.model_training.lstm_model import LSTM
from src.model_training.rnn_model import RNN
from src.model_training.gru_model import GRU
from src.model_training.transformer_encoder_model import TransformerEncoderRegressor


def build_sequence_model(model_type, input_dim, hidden_dim, num_layers, output_dim, dropout, **model_kwargs):
    model_type = str(model_type).lower()
    if model_type == "lstm":
        return LSTM(input_dim, hidden_dim, num_layers, output_dim, dropout)
    if model_type == "rnn":
        return RNN(input_dim, hidden_dim, num_layers, output_dim, dropout)
    if model_type == "gru":
        return GRU(input_dim, hidden_dim, num_layers, output_dim, dropout)
    if model_type == "transformer":
        return TransformerEncoderRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            nhead=int(model_kwargs.get("nhead", 4)),
            dim_feedforward=model_kwargs.get("dim_feedforward", None),
        )
    raise ValueError(f"Unsupported model_type: {model_type}. Use one of: lstm, rnn, gru, transformer")


def infer_model_type(model):
    if isinstance(model, LSTM):
        return "lstm"
    if isinstance(model, RNN):
        return "rnn"
    if isinstance(model, GRU):
        return "gru"
    if isinstance(model, TransformerEncoderRegressor):
        return "transformer"
    raise ValueError(
        "Unable to infer model type from instance. "
        "Please pass model_type explicitly (lstm/rnn/gru/transformer)."
    )


def get_model_arch_params(model, model_type=None):
    model_type = infer_model_type(model) if model_type is None else str(model_type).lower()

    if model_type == "lstm":
        hidden_dim = int(getattr(model, "hidden_dim", model.lstm.hidden_size))
        num_layers = int(getattr(model, "num_layers", model.lstm.num_layers))
        dropout = float(model.lstm.dropout)
    elif model_type == "rnn":
        hidden_dim = int(getattr(model, "hidden_dim", model.rnn.hidden_size))
        num_layers = int(getattr(model, "num_layers", model.rnn.num_layers))
        dropout = float(model.rnn.dropout)
    elif model_type == "gru":
        hidden_dim = int(getattr(model, "hidden_dim", model.gru.hidden_size))
        num_layers = int(getattr(model, "num_layers", model.gru.num_layers))
        dropout = float(model.gru.dropout)
    elif model_type == "transformer":
        hidden_dim = int(getattr(model, "hidden_dim", model.input_proj.out_features))
        num_layers = int(getattr(model, "num_layers", len(model.encoder.layers)))
        dropout = float(getattr(model, "dropout_p", model.dropout.p))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use one of: lstm, rnn, gru, transformer")

    params = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    if model_type == "transformer":
        if hasattr(model, "nhead"):
            params["nhead"] = int(model.nhead)
        else:
            params["nhead"] = int(model.encoder.layers[0].self_attn.num_heads)

        if hasattr(model, "dim_feedforward"):
            params["dim_feedforward"] = int(model.dim_feedforward) if model.dim_feedforward is not None else None
        else:
            params["dim_feedforward"] = int(model.encoder.layers[0].linear1.out_features)

    return params
