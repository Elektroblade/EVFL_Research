"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import numpy as np

from evfl.dnn import (
    DNN,
    INPUT_SIZE,
    OUTPUT_SIZE,
    HIDDEN_LAYERS,
    NEURONS_PER_LAYER,
    TASK_TYPE,
    PARTITION_SIZES,
    dnn_to_arrays,
    arrays_to_dnn,
    relu
)
from evfl.server_dnn import (
    ServerDNN
)

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]

    # ✅ Server owns ONLY its part of the model
    server = ServerDNN(
        input_size=sum(PARTITION_SIZES),   # sum of client output dims
        output_size=OUTPUT_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        learning_rate=lr,
        activation=relu,
        task_type=TASK_TYPE,
    )

    # Server owns labels
    trainloader = server.load_centralized_labels()

    for rnd in range(1, num_rounds + 1):
        print(f"\n[Server] Round {rnd}")

        for batch in trainloader:
            y = batch["y"]  # (output_size, batch_size)

            # ---- 1️⃣ Request forward activations ----
            results = grid.run(
                app_fn="forward",
                message=Message(
                    content=RecordDict({
                        "round": rnd
                    })
                ),
            )

            # ---- 2️⃣ Collect client activations ----
            client_activations = []
            client_dims = []

            for _, reply in results:
                h_i = reply.content["activations"]
                client_activations.append(h_i)
                client_dims.append(h_i.shape[0])

            # ---- 3️⃣ Concatenate activations ----
            h = np.vstack(client_activations)

            # ---- 4️⃣ Server forward + loss ----
            y_hat = server.forward(h)
            loss = server.compute_loss(y_hat, y)

            # ---- 5️⃣ Server backward ----
            grad_h = server.backward(y_hat, y)

            print(f"[Server] Loss: {loss:.4f}")

            # ---- 6️⃣ Split gradients per client ----
            grad_parts = []
            start = 0
            for dim in client_dims:
                grad_parts.append(grad_h[start:start + dim, :])
                start += dim

            # ---- 7️⃣ Send gradients back ----
            grid.run(
                app_fn="backward",
                message=Message(
                    content=RecordDict({
                        "grads": grad_parts
                    })
                ),
            )

    # ---- Save final server model ----
    print("\nSaving final NumPy ServerDNN model...")
    np.save("server_model.npy", server.to_numpy())



def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = DNN(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        learning_rate=0.0,  # no training here
        activation=relu,
        task_type=TASK_TYPE,
    )
    arrays_to_dnn(model, arrays)
    # attempt to move model to gpu here

    # Load entire test set
    test_dataloader = model.load_centralized_dataset()

    # Evaluate the global model on the test set
    preds, probs, y_true, avg_inf_ms = model.predict(test_dataloader)
    accuracy = np.mean(preds == y_true)

    return MetricRecord({
        "accuracy": float(accuracy),
        "avg_inference_time_ms": avg_inf_ms,
    })