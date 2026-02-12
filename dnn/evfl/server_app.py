"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from logging import INFO
from flwr.common import log
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

# ---------------------------------------------------------------------
# Create ServerApp
# ---------------------------------------------------------------------
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # ------------------------------------------------------------
    # Run config
    # ------------------------------------------------------------
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    batch_size: int = context.run_config["batch-size"]

    # ------------------------------------------------------------
    # Client embedding dimensions (fixed & known)
    # ------------------------------------------------------------
    client_embedding_dims = {
        f"client_{i}": dim for i, dim in enumerate(PARTITION_SIZES)
    }

    partition_id = msg.metadata["partition_id"]
    dim = client_embedding_dims[partition_id]

    # ------------------------------------------------------------
    # Server-side DNN head (NumPy)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Server owns labels ONLY
    # ------------------------------------------------------------
    trainloader, _ = server.load_centralized_labels(batch_size=batch_size)

    node_ids = list(grid.get_node_ids()) # TODO this causes problems
    log(INFO, "Connected clients: %s", node_ids)

    if len(node_ids) != len(client_embedding_dims):
        raise ValueError(
            f"Expected {len(client_embedding_dims)} clients, "
            f"but got {len(node_ids)}"
        )

    # ============================================================
    # Training loop
    # ============================================================
    for rnd in range(1, num_rounds + 1):
        log(INFO, "")
        log(INFO, "=== Server Round %s / %s ===", rnd, num_rounds)

        # NOTE:
        # This implementation assumes *aligned batches*
        # across clients and server (same sample indices).
        for batch_idx, batch in enumerate(trainloader):
            y = batch["y"]  # shape: (output_size, B)

            # ----------------------------------------------------
            # 1️⃣ Request embeddings from all clients
            # ----------------------------------------------------
            messages = []
            for pos, node_id in enumerate(node_ids):
                messages.append(
                    Message(
                        content=RecordDict({
                            "config": ConfigRecord({
                                "round": rnd,
                                "batch_idx": batch_idx,
                                "pos": pos,
                            }),
                        }),
                        message_type="query.forward",
                        dst_node_id=node_id,
                    )
                )

            log(INFO, "Requesting embeddings from %s clients", len(messages))
            replies = grid.send_and_receive(messages)

            # ----------------------------------------------------
            # 2️⃣ Assemble embedding matrix H
            # ----------------------------------------------------
            total_dim = sum(client_embedding_dims.values())
            batch_size = y.shape[1]
            H = np.zeros((total_dim, batch_size))

            node_pos_map: dict[str, int] = {}

            offset = 0
            for reply in replies:
                node_id = reply.metadata.src_node_id

                arr = reply.content["arrays"]["activations"]
                emb = arr.numpy()

                pos = reply.content["config"]["pos"]
                dim = client_embedding_dims[node_id]

                H[offset : offset + dim, :] = emb
                node_pos_map[node_id] = offset
                offset += dim

            # ----------------------------------------------------
            # 3️⃣ Server forward + backward (NumPy)
            # ----------------------------------------------------
            grads_w, grads_b, grad_H = server.backward(H, y)

            # ----------------------------------------------------
            # 4️⃣ Update server parameters (SGD)
            # ----------------------------------------------------
            for i in range(len(server.weights)):
                server.weights[i] -= lr * grads_w[i]
                server.biases[i] -= lr * grads_b[i]

            # ----------------------------------------------------
            # 5️⃣ Split embedding gradients per client
            # ----------------------------------------------------
            grad_per_client = {}
            for node_id, start in node_pos_map.items():
                dim = client_embedding_dims[node_id]
                grad_per_client[node_id] = grad_H[start : start + dim, :]

            # ----------------------------------------------------
            # 6️⃣ Send gradients back to clients
            # ----------------------------------------------------
            grad_messages = []
            for node_id, grad in grad_per_client.items():
                grad_messages.append(
                    Message(
                        content=RecordDict({
                            "arrays": ArrayRecord({
                                "grad": Array(grad),
                            }),
                        }),
                        message_type="train.backward",
                        dst_node_id=node_id,
                    )
                )

            log(INFO, "Sending gradients to %s clients", len(grad_messages))
            grid.push_messages(grad_messages)

        log(INFO, "Round %s complete", rnd)

    # ============================================================
    # Save final server model
    # ============================================================
    log(INFO, "")
    log(INFO, "Saving final ServerDNN model...")

    server_state = {
        "weights": server.weights,
        "biases": server.biases,
        "client_embedding_dims": server.client_embedding_dims,
        "task_type": server.task_type,
    }

    np.save("server_model.npy", server_state)
    log(INFO, "Model saved to server_model.npy")



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