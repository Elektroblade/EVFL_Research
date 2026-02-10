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
    DATASET_DIR,
    dnn_to_arrays,
    arrays_to_dnn,
    relu,
    relu_derivative
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

    # ---- Client embedding dimensions are known to the server ----
    # Example: {"client_0": 32, "client_1": 64, ...}
    client_embedding_dims = {
        f"client_{i}": dim
        for i, dim in enumerate(PARTITION_SIZES)
    }

    # ---- Server owns ONLY the top model + labels ----
    server = ServerDNN(
        client_embedding_dims=client_embedding_dims,
        output_size=OUTPUT_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        learning_rate=lr,
        activation=relu,
        activation_derivative=relu_derivative,
        task_type=TASK_TYPE,
    )

    # ---- Server owns labels ----
    trainloader, _ = server.load_centralized_labels(batch_size=context.run_config["batch-size"])

    for rnd in range(1, num_rounds + 1):
        print(f"\n[Server] Round {rnd}")

        for batch in trainloader:
            y = batch["y"]  # shape: (output_size, B)

            # ======================================================
            # 1️⃣ Request client forward passes
            # ======================================================
            results = grid.run(
                app_fn="forward",
                message=Message(
                    content = RecordDict({
                        "round": MetricRecord({"round": rnd}),
                    }),
                    dst_node_id=0,
                    message_type="forward",
                ),
            )

            # ======================================================
            # 2️⃣ Collect client embeddings (by client_id)
            # ======================================================
            client_embeddings = {}

            for client_id, reply in results:
                client_embeddings[client_id] = reply.content["activations"]

            # ======================================================
            # 3️⃣ Concatenate embeddings in fixed order
            # ======================================================
            ordered_embeddings = [
                client_embeddings[cid]
                for cid in client_embedding_dims.keys()
            ]
            H = np.vstack(ordered_embeddings)

            # ======================================================
            # 4️⃣ Server forward + backward
            # ======================================================
            grads_w, grads_b, grad_H = server.backward(H, y)

            # ---- Parameter update (simple SGD) ----
            for i in range(len(server.weights)):
                server.weights[i] -= lr * grads_w[i]
                server.biases[i] -= lr * grads_b[i]

            # ======================================================
            # 5️⃣ Split embedding gradients per client
            # ======================================================
            grad_per_client = server.split_embedding_gradients(grad_H)

            # ======================================================
            # 6️⃣ Send gradients back to clients
            # ======================================================
            for cid, grad in grad_per_client.items():
                grid.run(
                    app_fn="backward",
                    message=Message(
                        content=RecordDict({
                            "grad": ArrayRecord({"grad": grad}),
                        }),
                        dst_node_id=0,
                        message_type="backward",
                    ),
                    group_id=cid,
                )

        print("[Server] Round complete")

    # ==========================================================
    # 7️⃣ Save final server model
    # ==========================================================
    print("\n[Server] Saving final ServerDNN model...")

    server_state = {
        "weights": server.weights,
        "biases": server.biases,
        "client_embedding_dims": server.client_embedding_dims,
        "task_type": server.task_type,
    }

    np.save("server_model.npy", server_state)



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