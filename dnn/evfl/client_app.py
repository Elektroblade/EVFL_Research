"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord, Array
from flwr.clientapp import ClientApp
import numpy as np

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

    X_train, _ = model.load_data(
        partition_id,
        num_partitions,
        batch_size,
    )

    #print("X_train type during init:", type(X_train))

    return model, X_train  # materialize for deterministic indexing

def get_batch(loader, batch_idx):
    num_batches = len(loader)  # safe for PyTorch DataLoader
    effective_idx = batch_idx % num_batches

    for i, batch in enumerate(loader):
        if i == effective_idx:
            return batch

    raise IndexError("Batch index out of range")

# ------------------------------------------------------------------
# 1️⃣ Forward pass: generate embeddings
# ------------------------------------------------------------------
@app.query("forward")
def forward(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]
    batch_idx = msg.content["config"]["batch_idx"]
    node_id = context

    # --------------------------------------------------
    # First-time initialization
    # --------------------------------------------------
    if "model" not in context.state:

        model, X_train = _init_model_and_loader(context)

        # Store model parameters
        array_dict = {
            f"weights_{i}": Array(w.copy())
            for i, w in enumerate(model.weights)
        }
        array_dict.update({
            f"biases_{i}": Array(b.copy())
            for i, b in enumerate(model.biases)
        })

        context.state["model"] = ArrayRecord(array_dict)
    else:
        _, X_train = _init_model_and_loader(context)

    # --------------------------------------------------
    # Restore model from state
    # --------------------------------------------------
    stored = context.state["model"]

    num_layers = len([k for k in stored.keys() if "weights_" in k])

    weights = [
        stored[f"weights_{i}"].numpy()
        for i in range(num_layers)
    ]
    biases = [
        stored[f"biases_{i}"].numpy()
        for i in range(num_layers)
    ]

    model = ClientDNN(
        input_size=PARTITION_SIZES[partition_id],
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        activation=relu,
        activation_derivative=relu_derivative
    )

    model.weights = weights
    model.biases = biases

    # --------------------------------------------------
    # Deterministic batch slicing
    # --------------------------------------------------

    num_samples = X_train.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    effective_idx = batch_idx % num_batches

    start = effective_idx * batch_size
    end = min(start + batch_size, num_samples)

    # Slice rows (samples)
    X_batch = X_train[start:end]

    # Transpose to (features, batch_size)
    X_batch = X_batch.T

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    h = model.forward(X_batch)

    #print("Client embedding shape after forward:", h.shape)

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord({
                "activations": Array(h),
            }),
            "config": ConfigRecord({
                "pos": partition_id,
            }),
        }),
        reply_to=msg
    )


# ------------------------------------------------------------------
# 2️⃣ Backward pass: apply gradients
# ------------------------------------------------------------------
@app.train("backward")
def backward(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]
    batch_idx = msg.content["config"]["batch_idx"]  # must be sent from server

    # 1️⃣ Reload dataset
    model, X_train = _init_model_and_loader(context)

    #print("X_train type during backward:", type(X_train))

    # 2️⃣ Deterministic batch slicing (same as forward)
    num_samples = X_train.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    effective_idx = batch_idx % num_batches

    start = effective_idx * batch_size
    end = min(start + batch_size, num_samples)

    X_batch = X_train[start:end].T

    # 3️⃣ Restore model weights from state
    stored = context.state["model"]
    num_layers = len([k for k in stored.keys() if "weights_" in k])

    weights = [stored[f"weights_{i}"].numpy() for i in range(num_layers)]
    biases = [stored[f"biases_{i}"].numpy() for i in range(num_layers)]

    model.weights = weights
    model.biases = biases

    # 4️⃣ Recompute forward
    h = model.forward(X_batch)

    # 5️⃣ Get gradient from server
    grad_h = msg.content["arrays"]["grad"].numpy()

    # Safety check
    assert grad_h.shape == h.shape

    # 6️⃣ Backprop through client model
    grads_w, grads_b = model.backward(X_batch, grad_h)

    # 7️⃣ SGD update
    lr = context.run_config["learning-rate"]
    for i in range(len(model.weights)):
        model.weights[i] -= lr * grads_w[i]
        model.biases[i] -= lr * grads_b[i]

    # 8️⃣ Save updated weights back to state
    array_dict = {
        f"weights_{i}": Array(w.copy()) for i, w in enumerate(model.weights)
    }
    array_dict.update({
        f"biases_{i}": Array(b.copy()) for i, b in enumerate(model.biases)
    })

    context.state["model"] = ArrayRecord(array_dict)

    return Message(content=RecordDict(), reply_to=msg)