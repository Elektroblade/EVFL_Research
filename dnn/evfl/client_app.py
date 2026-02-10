"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord, Array
from flwr.clientapp import ClientApp

from evfl.dnn import (
    DNN,
    INPUT_SIZE,
    OUTPUT_SIZE,
    HIDDEN_LAYERS,
    NEURONS_PER_LAYER,
    PARTITION_SIZES,
    TASK_TYPE,
    DATASET_DIR,
    dnn_to_arrays,
    arrays_to_dnn,
    relu,
    relu_derivative
)
from evfl.client_dnn import (
    ClientDNN
)

# Flower ClientApp
app = ClientApp()

# ---- Globals ----
_batches_per_client: dict[int, list] = {}  # partition_id -> list of batches
# top-level global variable
_last_batch_x_per_client: dict[int, np.ndarray] = {}



def _init_model_and_loader(context: Context):
    """Initialize model and dataloader deterministically."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    model = ClientDNN(
        input_size=PARTITION_SIZES[partition_id],
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        activation=relu,
        activation_derivative=relu_derivative
    )

    trainloader, _ = model.load_data(
        partition_id,
        num_partitions,
        batch_size,
    )

    return model, list(trainloader)  # materialize for deterministic indexing


# ------------------------------------------------------------------
# 1️⃣ Forward pass: generate embeddings
# ------------------------------------------------------------------
@app.query("forward")
def forward(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]

    if "model" not in context.state:
        model, batches = _init_model_and_loader(context)

        # Store batches globally, keyed by partition_id
        _batches_per_client[partition_id] = batches

        # Store model weights/biases in ArrayRecord
        array_dict = {
            f"weights_{i}": Array(w.copy()) for i, w in enumerate(model.weights)
        }
        array_dict.update({
            f"biases_{i}": Array(b.copy()) for i, b in enumerate(model.biases)
        })
        context.state["model"] = ArrayRecord(array_dict)

    else:
        stored = context.state["model"]
        weights = [stored[f"weights_{i}"].numpy() for i in range(len(PARTITION_SIZES))]
        biases = [stored[f"biases_{i}"].numpy() for i in range(len(PARTITION_SIZES))]

        model = ClientDNN(
            input_size=PARTITION_SIZES[partition_id],
            hidden_layers=HIDDEN_LAYERS,
            neurons_per_layer=NEURONS_PER_LAYER,
            activation=relu,
            activation_derivative=relu_derivative
        )
        model.weights = weights
        model.biases = biases

        batches = _batches_per_client[partition_id]  # retrieve global batches


    # ---- Deterministic batch selection ----
    batch_idx = msg.content["config"]["batch_idx"]
    batch = batches[batch_idx % len(batches)]
    X = batch["x"]  # shape: (d_client, B)

    # Cache for backward
    partition_id = context.node_config["partition-id"]
    _last_batch_x_per_client[partition_id] = X

    # ---- Forward pass ----
    h = model.forward(X)  # shape: (d_embed, B)

    # Return activations in ArrayRecord
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord({
                "activations": Array(h),
            }),
            "config": ConfigRecord({
                "pos": partition_id,
            }),
        }),
        reply_to=msg,
    )


# ------------------------------------------------------------------
# 2️⃣ Backward pass: apply gradients
# ------------------------------------------------------------------
@app.train("backward")
def backward(msg: Message, context: Context) -> Message:
    model: ClientDNN = context.state["model"]
    partition_id = context.node_config["partition-id"]
    X = _last_batch_x_per_client[partition_id]

    grad_h = msg.content["arrays"]["grad"].numpy()  # (d_embed, B)

    # Backward pass
    grads_w, grads_b = model.backward(X, grad_h)

    # SGD update
    lr = context.run_config["learning-rate"]
    for i in range(len(model.weights)):
        model.weights[i] -= lr * grads_w[i]
        model.biases[i] -= lr * grads_b[i]

    # Persist model
    # Store model weights/biases in ArrayRecord
    array_dict = {
        f"weights_{i}": Array(w.copy()) for i, w in enumerate(model.weights)
    }
    array_dict.update({
        f"biases_{i}": Array(b.copy()) for i, b in enumerate(model.biases)
    })
    context.state["model"] = ArrayRecord(array_dict)

    return Message(content=RecordDict(), reply_to=msg)