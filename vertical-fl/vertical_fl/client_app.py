import io

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from vertical_fl.task import ClientModel, load_data
from logging import INFO
from flwr.common import log
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Flower ClientApp
app = ClientApp()


def client_load_data(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    #global_batch_size = context.run_config["batch-size"]
    subset_size = context.run_config["subset"]

    feature_splits: str = context.run_config["feature-splits"]
    in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]

    log(INFO, f"Loading data for p{partition_id}...")
    X_train, X_test = load_data(partition_id, in_feature_dim_clientapp, subset_size)

    print(type(X_train), type(X_test))
    print(isinstance(X_train, np.ndarray), isinstance(X_test, np.ndarray))

    log(INFO, f"Loaded data for p{partition_id}")

    context.state["data"] = ArrayRecord([X_train, X_test], keep_input=True)


@app.query("generate_embeddings")
def get_emb(msg: Message, context: Context):
    """Generate embeddings."""

    # Read from config
    partition_id = context.node_config["partition-id"]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]
    mode = msg.content["config"]["mode"]
    if "data" not in context.state:
        client_load_data(context)

    # determine whether to use training or testing data
    if mode == -1:
        data = context.state["data"].to_numpy_ndarrays()[1]
    else:
        data = context.state["data"].to_numpy_ndarrays()[0]

    data = torch.from_numpy(data).float()
    model = ClientModel(
        input_size=data.shape[1], out_feat_dim=out_feature_dim_clientapp
    )

    # Load model from state if available
    if model_record := context.state.get("model", None):
        model.load_state_dict(model_record.to_torch_state_dict())

    # Do forward pass
    embedding = model(data)

    # Construct and return reply Message
    model_record = ArrayRecord({"embedding": Array(embedding.detach().numpy())})
    content = RecordDict(
        {
            "arrays": model_record,
            "config": ConfigRecord({"pos": partition_id}),
        }
    )
    return Message(content=content, reply_to=msg)


@app.train("apply_gradients")
def apply_grad(msg: Message, context: Context):
    """Apply gradients to local model."""

    # Read from config
    lr = context.run_config["learning-rate"]
    partition_id = context.node_config["partition-id"]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]
    mode = msg.content["config"]["mode"]
    if "data" not in context.state:
        load_data(context)

    # determine whether to use training or testing data
    if mode == -1:
        data = context.state["data"].to_numpy_ndarrays()[1]
    else:
        data = context.state["data"].to_numpy_ndarrays()[0]

    data = torch.from_numpy(data).float()
    model = ClientModel(
        input_size=data.shape[1], out_feat_dim=out_feature_dim_clientapp
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # Load model from state if available
    if model_record := context.state.get("model", None):
        model.load_state_dict(model_record.to_torch_state_dict())

    # Load optimizer state from state if available
    if optimizer_state_record := context.state.get("optimizer_state", None):
        buffer = io.BytesIO(optimizer_state_record["serialized"])
        optimizer.load_state_dict(torch.load(buffer))

    # Do forward pass
    embedding = model(data)

    # Get gradients from message and apply them
    embedding.backward(
        torch.from_numpy(msg.content["gradients"]["local-gradients"].numpy())
    )
    optimizer.step()

    # Save updated model in state for next round
    context.state["model"] = ArrayRecord(model.state_dict())
    # (Optional) Save the optimizer state. Not all optimizers have state, but Adam does.
    torch.save(optimizer.state_dict(), buffer := io.BytesIO())
    context.state["optimizer_state"] = ConfigRecord({"serialized": buffer.getvalue()})

    # Construct and return reply Message
    return Message(content=RecordDict(), reply_to=msg)
