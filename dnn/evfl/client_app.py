"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
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

# ---- Global model (persistent across rounds) ----
client_model: ClientDNN | None = None
client_dataloader = None


def start_client_app(context: Context):
    """Initialize client model and data loader"""
    global client_model, client_dataloader, last_batch_x

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    # ---- Initialize client model (feature-only) ----
    client_model = ClientDNN(
        input_size=PARTITION_SIZES[partition_id],
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        activation=relu,
        activation_derivative=relu_derivative
    )

    # ---- Client owns only its data partition ----
    trainloader, _ = client_model.load_data(
        partition_id,
        num_partitions,
        batch_size,
    )
    client_dataloader = iter(trainloader)

    # Cache for backward pass alignment
    last_batch_x = None


def forward(msg: Message, context: Context):
    global client_model, client_dataloader, last_batch_x

    try:
        batch = next(client_dataloader)
    except StopIteration:
        # Restart epoch deterministically
        client_dataloader = iter(
            client_model.load_data(
                context.node_config["partition-id"],
                context.node_config["num-partitions"],
                context.run_config["batch-size"],
            )
        )
        batch = next(client_dataloader)

    X = batch["x"]   # shape: (d_client, B)
    last_batch_x = X

    # ---- Client forward pass ----
    h = client_model.forward(X)

    return Message(
        content=RecordDict({
            "activations": h
        }),
        reply_to=msg,
    )


def backward(msg: Message, context: Context):
    global client_model, last_batch_x

    # ---- Gradient w.r.t. client embedding ----
    grad_h = msg.content["grads"]  # shape: (h_dim, B)

    # ---- Client backward pass ----
    grads_w, grads_b = client_model.backward(
        last_batch_x,
        grad_h
    )

    # ---- Client-side parameter update (SGD) ----
    lr = context.run_config["learning-rate"]
    for i in range(len(client_model.weights)):
        client_model.weights[i] -= lr * grads_w[i]
        client_model.biases[i] -= lr * grads_b[i]

    return Message(
        content=RecordDict({}),
        reply_to=msg,
    )



"""
@app.evaluate()
def evaluate(msg: Message, context: Context):
    Evaluate the model on local data.

    # Load the model and initialize it with the received weights
    model = DNN(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        learning_rate=0.0,  # no training
        activation=relu,
        task_type=TASK_TYPE,
    )
    arrays_to_dnn(model, msg.content["arrays"])

    # Load the data
    partition_id = context.node_id
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = model.load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    preds, probs, y_true, avg_inf_ms = model.predict(valloader)

    accuracy = float((preds == y_true).mean())

    # Construct and return reply Message
    metrics = {
        "eval_acc": accuracy,
        "avg_inference_time_ms": avg_inf_ms,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

"""