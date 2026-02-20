"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, MetricRecord, Message, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from logging import INFO
from flwr.common import log
import numpy as np
from collections import defaultdict
import time
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import gc
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)
import pandas as pd

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
    relu_derivative,
    softmax
)
from evfl.server_dnn import (
    ServerDNN
)
RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# ---------------------------------------------------------------------
# Create ServerApp
# ---------------------------------------------------------------------
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Run config
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    global_batch_size: int = context.run_config["batch-size"]
    subset_size: int = context.run_config["subset"]

    # Client embedding dimensions (fixed & known)
    embedding_dim = NEURONS_PER_LAYER
    num_clients = len(PARTITION_SIZES)

    # Server-side DNN head (NumPy)
    server = ServerDNN(
        num_clients=num_clients,
        embedding_dim=embedding_dim,
        output_size=OUTPUT_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        learning_rate=lr,
        activation=relu,
        activation_derivative=relu_derivative,
        task_type=TASK_TYPE,
    )

    print("Server output size:", server.output_size)

    # ------------------------------------------------------------
    # Server owns labels ONLY
    # ------------------------------------------------------------
    trainloader, testloader, total_number_of_samples = server.load_centralized_labels(global_batch_size, subset_size)

    node_ids = list(grid.get_node_ids())
    log(INFO, "Connected clients: %s", node_ids)

    client_embedding_dims = {
        node_id: NEURONS_PER_LAYER
        for node_id in node_ids
    }

    if len(node_ids) != len(client_embedding_dims):
        raise ValueError(
            f"Expected {len(client_embedding_dims)} clients, "
            f"but got {len(node_ids)}"
        )

    server, training_history = train(grid, context, num_rounds, lr, embedding_dim, num_clients,
          server, trainloader, total_number_of_samples, node_ids, global_batch_size)
    
    os.makedirs(os.path.dirname("./server_model/"), exist_ok=True)
    # Save to disk
    np.savez(
        f"./server_model/training_history_dnn_vfl_{subset_size}sa_{num_rounds}eps.npz",
        predictions=training_history["predictions"],
        prediction_probs=training_history["prediction_probs"],
        real_values=training_history["real_values"],
        avg_inference_time_ms=training_history["avg_inference_time_ms"],
    )

    train_metrics_single = save_test_metrics_single(num_rounds, f"dnn_vfl_{subset_size}sa", "server_model", 
        [0, 1, 2, 3, 4, 5, 6, 7], subset_size, mode=0)

    # Save final server model
    log(INFO, "")
    log(INFO, "Saving final ServerDNN model...")

    server.client_embedding_dims = {
        f"client_{i}": 32 for i in range(len(PARTITION_SIZES))
    }

    server_state = {
        "weights": server.weights,
        "biases": server.biases,
        "client_embedding_dims": server.client_embedding_dims,
        "task_type": server.task_type,
    }

    np.save(f"./server_model/dnn_vfl_{subset_size}sa_{num_rounds}eps.npy", server_state)
    log(INFO, "Model saved to server_model.npy")

    prediction_history = test(grid, context, num_rounds, lr, embedding_dim, num_clients,
          server, testloader, total_number_of_samples, node_ids, global_batch_size)
    
    # Save to disk
    np.savez(
        f"./server_model/prediction_history_dnn_vfl_{subset_size}sa_{num_rounds}eps.npz",
        predictions=prediction_history["predictions"],
        prediction_probs=prediction_history["prediction_probs"],
        real_values=prediction_history["real_values"],
        avg_inference_time_ms=prediction_history["avg_inference_time_ms"],
    )

    test_metrics_single = save_test_metrics_single(num_rounds, f"dnn_vfl_{subset_size}sa", "server_model", 
        [0, 1, 2, 3, 4, 5, 6, 7], subset_size, mode=-1)
    

def train(grid, context, num_rounds, lr, embedding_dim, num_clients,
          server, trainloader, total_number_of_samples, node_ids, global_batch_size):
    # Training loop
    training_history = defaultdict(list)
    total_inference_time_ms = 0
    processed_samples = 0
    log(INFO, "STARTING TRAINING")

    for rnd in range(1, num_rounds + 1):
        num_samples = trainloader.shape[1]

        training_history["predictions"].append([])
        training_history["prediction_probs"].append([])
        training_history["real_values"].append([])

        log(INFO, "")
        log(INFO, "=== Server Round %s / %s ===", rnd, num_rounds)

        # NOTE:
        # This implementation assumes *aligned batches*
        # across clients and server (same sample indices).
        for batch_count, start in enumerate(range(0, num_samples, global_batch_size)):
            end = min(start + global_batch_size, num_samples)
            y = trainloader[:, start:end]  # shape: (output_size, B)

            #print("Server train Unique y values:", np.unique(y))

            effective_batch_size = y.shape[1]
            processed = min(end, num_samples)
            log(INFO, f"eps: {rnd}, bi: {batch_count}, processed: {processed} / {total_number_of_samples} samples")

            # 1 Request embeddings from all clients
            messages = []
            for pos, node_id in enumerate(node_ids):
                messages.append(
                    Message(
                        content=RecordDict({
                            "config": ConfigRecord({
                                "round": rnd,
                                "batch_idx": batch_count,
                                "pos": pos,
                                "global_batch_size": global_batch_size,
                                "effective_batch_size": effective_batch_size,  # <--- pass correct batch size
                                "mode": 0
                            }),
                        }),
                        message_type="query.forward",
                        dst_node_id=node_id,
                    )
                )

            t0 = time.time()
            #log(INFO, "Requesting embeddings from %s clients", len(messages))
            replies = grid.send_and_receive(messages)
            #log(INFO, "Received embeddings from %s clients", len(messages))

            # 2 Assemble embedding matrix H
            embedding_dim = NEURONS_PER_LAYER
            num_clients = len(node_ids)
            effective_batch_size = y.shape[1]

            total_dim = embedding_dim * num_clients
            H = np.zeros((total_dim, effective_batch_size))

            node_pos_map: dict[int, tuple[int, int]] = {}

            # --- DEBUG: Check client embeddings vs server labels ---
            #DEBUG_NUM_SAMPLES = min(5, effective_batch_size)  # first few samples

            for i, reply in enumerate(replies):
                node_id = reply.metadata.src_node_id

                emb = reply.content["arrays"]["activations"].numpy()

                #print(f"[Client {i}] Embedding mean: {emb.mean():.6f}, std: {emb.std():.6f}, min: {emb.min():.6f}, max: {emb.max():.6f}")

                """
                print(f"\n--- Client {node_id} Embedding Debug ---")
                print(f"Embedding shape: {emb.shape}")

                for sample_idx in range(DEBUG_NUM_SAMPLES):
                    emb_vec = emb[:, sample_idx]
                    server_label = y[:, sample_idx]  # shape: (output_size,)
                    print(f"Sample {sample_idx}: Embedding (first 5 values) {emb_vec[:5]} | Server label: {server_label.flatten()}")
                """

                start = i * embedding_dim
                end = start + embedding_dim

                H[start:end, :] = emb
                node_pos_map[node_id] = (start, embedding_dim)

            # 3 Server forward + backward (NumPy)
            grads_w, grads_b, grad_H, predictions_probs = server.backward(H, y)

            # 4 Update server parameters (SGD)
            for i in range(len(server.weights)):
                server.weights[i] -= lr * grads_w[i]
                server.biases[i] -= lr * grads_b[i]
            
            t1 = time.time()
            batch_time_ms = (t1 - t0) * 1000
            batch_time_ms = (t1 - t0) * 1000

            total_inference_time_ms += batch_time_ms
            processed_samples += effective_batch_size

            # Convert to predicted classes if multiclass
            if server.task_type == "multiclass":
                #print("Output shape:", predictions_probs.shape)
                predictions = np.argmax(predictions_probs, axis=0)
            elif server.task_type == "binary":
                predictions = (predictions_probs > 0.5).astype(int)
            else:
                predictions = predictions_probs  # regression

            # 5 Split embedding gradients per client
            grad_per_client = {}
            for node_id, (start, dim) in node_pos_map.items():
                grad_per_client[node_id] = grad_H[start : start + dim, :]

            # 6 Send gradients back to clients
            grad_messages = []
            for node_id, grad in grad_per_client.items():
                grad_messages.append(
                    Message(
                        content=RecordDict({
                            "arrays": ArrayRecord({
                                "grad": Array(grad),
                            }),
                            "config": ConfigRecord({
                                "global_batch_size": global_batch_size,
                                "effective_batch_size": effective_batch_size,
                                "batch_idx": batch_count,
                                "mode": 0
                            }),
                        }),
                        message_type="train.backward",
                        dst_node_id=node_id,
                    )
                )

            #log(INFO, "Sending gradients to %s clients", len(grad_messages))
            grid.push_messages(grad_messages)

            # 4 Store history
            training_history["predictions"][rnd-1].extend(predictions.flatten())
            training_history["prediction_probs"][rnd-1].extend(predictions_probs.flatten())
            if server.task_type == "multiclass":
                true_labels = np.argmax(y, axis=0)
            elif server.task_type == "binary":
                true_labels = y.flatten()
            else:
                true_labels = y.flatten()

            training_history["real_values"][rnd-1].extend(true_labels)

            del grad_H
            del H
            gc.collect()  # force Python garbage collection

        log(INFO, "Round %s complete", rnd)
        print("Total samples:", processed_samples)

    training_history["predictions"] = np.stack(
        training_history["predictions"], axis=0
    ).astype(np.int32)
    training_history["prediction_probs"] = np.stack(
        training_history["prediction_probs"], axis=0
    ).astype(np.int32)
    training_history["real_values"] = np.stack(
        training_history["real_values"], axis=0
    ).astype(np.int32)

    # avg_inference_time_ms: keep as list or average over batches
    avg_per_sample_time_ms = total_inference_time_ms / processed_samples

    training_history["avg_inference_time_ms"] = np.array([avg_per_sample_time_ms]).astype(np.int32)
    return server, training_history


def test(grid, context, num_rounds, lr, embedding_dim, num_clients,
          server, testloader, total_number_of_samples, node_ids, global_batch_size):
    prediction_history = defaultdict(list)
    total_inference_time_ms = 0
    processed_samples = 0

    num_samples = testloader.shape[1]
    # Testing loop
    log(INFO, "STARTING TESTING")

    for batch_count, start in enumerate(range(0, num_samples, global_batch_size)):
        end = min(start + global_batch_size, num_samples)
        y = testloader[:, start:end]  # shape: (N,)
        effective_batch_size = y.shape[1]
        processed = min(end, total_number_of_samples)
        print(f"bi: {batch_count}, processed: {processed} / {total_number_of_samples} samples")

        # 1 Request embeddings from all clients
        messages = []
        for pos, node_id in enumerate(node_ids):
            messages.append(
                Message(
                    content=RecordDict({
                        "config": ConfigRecord({
                            "round": 0,
                            "batch_idx": batch_count,
                            "pos": pos,
                            "global_batch_size": global_batch_size,
                            "effective_batch_size": effective_batch_size,  # <--- pass correct batch size
                            "mode": -1
                        }),
                    }),
                    message_type="query.forward",
                    dst_node_id=node_id,
                )
            )

        #log(INFO, "Requesting embeddings from %s clients", len(messages))

        t0 = time.time()
        replies = grid.send_and_receive(messages)

        # 2 Assemble embedding matrix H
        embedding_dim = NEURONS_PER_LAYER
        num_clients = len(node_ids)
        effective_batch_size = y.shape[1]

        total_dim = embedding_dim * num_clients
        H = np.zeros((total_dim, effective_batch_size))

        for i, reply in enumerate(replies):
            node_id = reply.metadata.src_node_id
            emb = reply.content["arrays"]["activations"].numpy()
            start_row = i * embedding_dim
            end_row = start_row + embedding_dim
            H[start_row:end_row, :] = emb

        # 3 Server forward (no backward)
        predictions_probs, _, _ = server.forward(H)

        t1 = time.time()
        batch_time_ms = (t1 - t0) * 1000
        batch_time_ms = (t1 - t0) * 1000

        total_inference_time_ms += batch_time_ms
        processed_samples += effective_batch_size

        # Convert to predicted classes if multiclass
        if server.task_type == "multiclass":
            #print("Output shape:", predictions_probs.shape)
            predictions = np.argmax(predictions_probs, axis=0)
        elif server.task_type == "binary":
            predictions = (predictions_probs > 0.5).astype(int)
        else:
            predictions = predictions_probs  # regression

        # 4 Store history
        prediction_history["predictions"].extend(predictions.flatten())
        prediction_history["prediction_probs"].extend(predictions_probs.flatten())
        if server.task_type == "multiclass":
            true_labels = np.argmax(y, axis=0)
        elif server.task_type == "binary":
            true_labels = y.flatten()
        else:
            true_labels = y.flatten()

        prediction_history["real_values"].extend(true_labels)

    prediction_history["predictions"] = np.array(prediction_history["predictions"])
    prediction_history["prediction_probs"] = np.array(prediction_history["prediction_probs"])
    prediction_history["real_values"] = np.array(prediction_history["real_values"])

    # avg_inference_time_ms: keep as list or average over batches
    avg_per_sample_time_ms = total_inference_time_ms / processed_samples

    prediction_history["avg_inference_time_ms"] = np.array([avg_per_sample_time_ms])

    return prediction_history

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

def save_test_metrics_single(num_epochs, 
    model_name, model_directory, TARGET_LIST, subset_size, confusion_matrix_fig_size=(10, 8), mode = -1
):
    """
    Uses testing predictions stored in prediction_history (NumPy arrays) to build
    a confusion matrix and compute test metrics. Saves metrics to disk.
    
    Args:
        model_name (str): Name of the model.
        model_directory (str): Directory to save metrics/plots.
        TARGET_LIST (list): List of all possible target classes.
        prediction_history (dict): Output of your test() function with keys:
            - "predictions" -> np.ndarray, shape (output_size, num_samples)
            - "prediction_probs" -> np.ndarray, shape (num_classes, num_samples)
            - "real_values" -> np.ndarray, shape (output_size, num_samples)
            - "avg_inference_time_ms" -> float
        confusion_matrix_fig_size (tuple): Figure size for confusion matrix plot.
    
    Returns:
        test_metrics (defaultdict): Dictionary of aggregated metrics.
    """

    if mode == -1:
        pred_prefix = "prediction"
        metric_prefix = "test"
    else:
        pred_prefix = "training"
        metric_prefix = "train"

    os.makedirs(model_directory, exist_ok=True)

    prediction_history = np.load(f"./server_model/{pred_prefix}_history_dnn_vfl_{subset_size}sa_{num_epochs}eps.npz", allow_pickle=True)

    if mode == -1:
        if num_epochs > 0:
            confusion_matrix_file_name = f"{metric_prefix}_cm_{model_name}_{num_epochs}eps"
        else:
            confusion_matrix_file_name = f"{metric_prefix}_cm_{model_name}"
        y_pred = prediction_history["predictions"]
        y_proba = prediction_history["prediction_probs"]
        real_values = prediction_history["real_values"]
        inference_time = prediction_history["avg_inference_time_ms"][0]

        # Flatten if necessary (shape: num_samples,)
        if y_pred.ndim > 1 and y_pred.shape[0] == 1:
            y_pred = y_pred.flatten()
        if real_values.ndim > 1 and real_values.shape[0] == 1:
            real_values = real_values.flatten()
        
        # Confusion matrix
        print("\n--- DEBUG LABEL INSPECTION ---")

        print("real_values shape:", real_values.shape)
        print("y_pred shape:", y_pred.shape)

        print("Unique real_values:", np.unique(real_values))
        print("Unique y_pred:", np.unique(y_pred))

        present_labels = sorted(set(real_values) | set(y_pred))
        print("present_labels:", present_labels)

        valid_labels = [label for label in TARGET_LIST if label in present_labels]
        print("valid_labels:", valid_labels)

        print("--- END DEBUG ---\n")

        cm = confusion_matrix(real_values, y_pred, labels=valid_labels)

        def format_k(x):
            return f"{x/1000:.1f}k" if x >= 1000 else str(x)

        # Create formatted annotations
        annot = np.array([[format_k(val) for val in row] for row in cm])

        df_cm = pd.DataFrame(cm, index=valid_labels, columns=valid_labels)
        os.makedirs(os.path.dirname("./figures/"), exist_ok=True)
        show_confusion_matrix(df_cm, annot, confusion_matrix_fig_size, confusion_matrix_file_name, num_epochs, mode=mode)

        # Compute metrics
        test_metrics = defaultdict(float)

        # Micro accuracy
        test_metrics["micro_acc"] = accuracy_score(real_values, y_pred)

        # Binarize for ROC-AUC
        n_classes = len(np.unique(real_values))
        y_true_bin = label_binarize(real_values, classes=np.arange(n_classes))

        # Micro ROC-AUC
        try:
            test_metrics["micro_roc_auc"] = roc_auc_score(
                y_true_bin, y_proba.T, average="micro", multi_class="ovr"
            )
        except ValueError:
            test_metrics["micro_roc_auc"] = float("nan")

        # Macro metrics
        test_metrics["macro_prec"] = precision_score(real_values, y_pred, average="macro", zero_division=0)
        test_metrics["macro_rec"] = recall_score(real_values, y_pred, average="macro", zero_division=0)
        test_metrics["macro_f1"] = f1_score(real_values, y_pred, average="macro")

        try:
            test_metrics["macro_roc_auc"] = roc_auc_score(
                y_true_bin, y_proba.T, average="macro", multi_class="ovr"
            )
        except ValueError:
            test_metrics["macro_roc_auc"] = float("nan")

        # Inference time
        test_metrics["inference_time"] = inference_time

        # Save metrics
        metrics_path = os.path.join(model_directory, f"{metric_prefix}_metrics_{model_name}.npz")
        np.savez(metrics_path, **test_metrics)

        print(f"[Info] Saved {metric_prefix} metrics to {metrics_path}")

        plot_test_metrics_table(
            test_metrics,
            model_name=model_name,
            num_epochs=num_epochs,
            mode=mode
        )
    else:
        for rnd_idx, y_pred in enumerate(prediction_history["predictions"]):
            if num_epochs > 0:
                confusion_matrix_file_name = f"{metric_prefix}_cm_{model_name}_{rnd_idx+1}eps"
            else:
                confusion_matrix_file_name = f"{metric_prefix}_cm_{model_name}"
            y_proba = prediction_history["prediction_probs"][rnd_idx]
            real_values = prediction_history["real_values"][rnd_idx]
            inference_time = prediction_history["avg_inference_time_ms"][0]

            # Flatten if necessary (shape: num_samples,)
            if y_pred.ndim > 1 and y_pred.shape[0] == 1:
                y_pred = y_pred.flatten()
            if real_values.ndim > 1 and real_values.shape[0] == 1:
                real_values = real_values.flatten()
            
            # Confusion matrix
            print("\n--- DEBUG LABEL INSPECTION ---")

            print("real_values shape:", real_values.shape)
            print("y_pred shape:", y_pred.shape)

            print("Unique real_values:", np.unique(real_values))
            print("Unique y_pred:", np.unique(y_pred))

            present_labels = sorted(set(real_values) | set(y_pred))
            print("present_labels:", present_labels)

            valid_labels = [label for label in TARGET_LIST if label in present_labels]
            print("valid_labels:", valid_labels)

            print("--- END DEBUG ---\n")

            cm = confusion_matrix(real_values, y_pred, labels=valid_labels)

            def format_k(x):
                return f"{x/1000:.1f}k" if x >= 1000 else str(x)

            # Create formatted annotations
            annot = np.array([[format_k(val) for val in row] for row in cm])

            df_cm = pd.DataFrame(cm, index=valid_labels, columns=valid_labels)
            os.makedirs(os.path.dirname("./figures/"), exist_ok=True)
            show_confusion_matrix(df_cm, annot, confusion_matrix_fig_size, confusion_matrix_file_name, num_epochs, mode=mode)

            # Compute metrics
            test_metrics = defaultdict(float)

            # Micro accuracy
            test_metrics["micro_acc"] = accuracy_score(real_values, y_pred)

            # Binarize for ROC-AUC
            n_classes = len(np.unique(real_values))
            y_true_bin = label_binarize(real_values, classes=np.arange(n_classes))

            # Micro ROC-AUC
            try:
                test_metrics["micro_roc_auc"] = roc_auc_score(
                    y_true_bin, y_proba.T, average="micro", multi_class="ovr"
                )
            except ValueError:
                test_metrics["micro_roc_auc"] = float("nan")

            # Macro metrics
            test_metrics["macro_prec"] = precision_score(real_values, y_pred, average="macro", zero_division=0)
            test_metrics["macro_rec"] = recall_score(real_values, y_pred, average="macro", zero_division=0)
            test_metrics["macro_f1"] = f1_score(real_values, y_pred, average="macro")

            try:
                test_metrics["macro_roc_auc"] = roc_auc_score(
                    y_true_bin, y_proba.T, average="macro", multi_class="ovr"
                )
            except ValueError:
                test_metrics["macro_roc_auc"] = float("nan")

            # Inference time
            test_metrics["inference_time"] = inference_time

            # Save metrics
            metrics_path = os.path.join(model_directory, f"{metric_prefix}_metrics_{model_name}_{rnd_idx+1}eps.npz")
            np.savez(metrics_path, **test_metrics)

            print(f"[Info] Saved {metric_prefix} metrics to {metrics_path}")

            plot_test_metrics_table(
                test_metrics,
                model_name=model_name,
                num_epochs=rnd_idx+1,
                mode=mode
            )


    return test_metrics

def show_confusion_matrix(confusion_matrix, annot, confusion_matrix_fig_size, figure_version, num_epochs, mode=-1):
    plt.figure(figsize=confusion_matrix_fig_size)
    sns.heatmap(confusion_matrix, annot=annot, cmap='Blues', fmt='')
    plt.xticks(rotation=90)
    plt.ylabel('Real threats')
    plt.xlabel('Predicted threats')
    plt.savefig(f'./figures/{figure_version}.png',bbox_inches="tight",dpi=300)
    plt.clf()

def plot_test_metrics_table(test_metrics, model_name, num_epochs, mode=-1):
    """
    Plots and saves a table of test metrics as a figure.

    Parameters:
    - test_metrics: defaultdict or dict with scalar values
    - model_name: str (e.g., 'bert_mini')
    - num_epochs: int
    """

    # Define a mapping from metric keys to display names
    metric_name_map = {
        "micro_acc": "Overall Accuracy",
        "micro_rec": "Overall Recall",
        "micro_prec": "Overall Precision",
        "micro_f1": "Overall F1 Score",
        "micro_roc_auc": "Overall ROC AUC",
        "macro_acc": "Class-Averaged Accuracy",
        "macro_rec": "Class-Averaged Recall",
        "macro_prec": "Class-Averaged Precision",
        "macro_f1": "Class-Averaged F1 Score",
        "macro_roc_auc": "Class-Averaged ROC AUC",
        "training_time": "Classification Training Time (h)",
        "inference_time": "Inference Latency (ms)"
        # Add more mappings as needed
    }

    if mode == -1:
        metric_prefix = "test"
    else:
        metric_prefix = "train"

    # Convert metrics to a list of [metric_name, value] rows
    data = []
    for key, value in test_metrics.items():
        label = metric_name_map.get(key, key)  # Use key as fallback if not in map
        if (value > -100.0):
            data.append([label, f"{value:.4f}"])
        else:
            data.append([label, "-"])

    fig, ax = plt.subplots(figsize=(8, len(data) * 0.4 + 1))
    ax.axis('off')

    table = ax.table(
        cellText=data,
        colLabels=["Metric", "Score"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    output_path = f"./figures/{metric_prefix}_scores_{model_name}_{num_epochs}eps.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.clf()