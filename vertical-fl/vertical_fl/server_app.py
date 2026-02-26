from collections import defaultdict
from logging import INFO
import numpy as np
import pandas as pd
from sklearn.calibration import label_binarize
import torch
from datasets import load_dataset
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from datasets import Dataset
from datasets import DatasetDict
import time
import random
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_from_disk
from logging import INFO
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import gc
import torch.nn as nn
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

from vertical_fl.task import (FEATURE_COLUMNS, 
    TARGET_COLUMN, DATASET_DIR, SEED, ServerModel, evaluate_head_model, TASK_TYPE, OUTPUT_SIZE, PARTITION_SIZES, DATASET_NAME, MODEL_FAMILY)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    subset_size: int = context.run_config["subset"]
    num_rounds: int = context.run_config["num-server-rounds"]
    feature_splits: str = context.run_config["feature-splits"]
    in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]
    if sum(in_feature_dim_clientapp) != len(FEATURE_COLUMNS):
        raise ValueError(
            "The sum of feature splits must equal the total number of features "
            f"(got {sum(in_feature_dim_clientapp)} vs. {len(FEATURE_COLUMNS)})."
        )
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]

    # Get dataset
    dataset = load_from_disk(DATASET_DIR)

    # Currently the entire dataset is in a pseudo "train" split
    dataset = dataset["train"].train_test_split(
        test_size=0.2,
        seed=SEED,
    )

    if 0 < subset_size < len(dataset["train"]):
        train_dataset = dataset["train"].select(range(subset_size))
        test_dataset = dataset["test"].select(range(subset_size))
    else:
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    
    head = train(train_dataset, num_rounds, grid, in_feature_dim_clientapp, out_feature_dim_clientapp)

    os.makedirs(os.path.dirname("./server_model/"), exist_ok=True)
    model_name = f"{DATASET_NAME}_{MODEL_FAMILY}_vfl_{subset_size}sa_{num_rounds}eps"

    # Save final server model
    log(INFO, "")
    log(INFO, "Saving final ServerDNN model...")

    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]
    head.client_embedding_dims = {
        f"client_{i}": out_feature_dim_clientapp for i in range(len(PARTITION_SIZES))
    }

    torch.save(head.state_dict(), f"./server_model/{model_name}_state.pt")
    metadata = {
        "input_size": head.input_size,
        "num_classes": head.num_classes,
        "client_embedding_dims": head.client_embedding_dims,
        "task_type": TASK_TYPE,
        "dropout_rate": getattr(head, "dropout_rate", None),
        "hidden_layers": [
            layer.out_features
            for name, layer in head.named_modules()
            if isinstance(layer, nn.Linear) and name.startswith("hidden")
        ]
    }
    np.save(f"./server_model/{model_name}_metadata.npy", metadata)

    metadata = np.load(
        f"./server_model/{model_name}_metadata.npy",
        allow_pickle=True
    ).item()

    print(metadata)

    if TASK_TYPE == "binary":
        head = ServerModel(input_size=metadata["input_size"])
    else:  # multiclass
        head = ServerModel(input_size=metadata["input_size"], num_classes=metadata["num_classes"])

    testing_history = test(test_dataset, num_rounds, grid, in_feature_dim_clientapp, out_feature_dim_clientapp, head)

    # Save to disk
    np.savez(
        f"./server_model/testing_history_{model_name}.npz",
        predictions=testing_history["predictions"],
        prediction_probs=testing_history["prediction_probs"],
        probs=testing_history["probs"],
        real_values=testing_history["real_values"],
        avg_inference_time_ms=testing_history["avg_inference_time_ms"],
    )

    test_metrics_single = save_test_metrics_single(num_rounds, model_name, "server_model", 
        [0, 1, 2, 3, 4, 5, 6, 7], subset_size, mode=-1)


def train(dataset, num_rounds, grid, in_feature_dim_clientapp, out_feature_dim_clientapp):
    labels = dataset["target"]
    
    # ----- TASK-SPECIFIC LABEL HANDLING -----
    if TASK_TYPE == "binary":
        labels = torch.tensor(labels).float().unsqueeze(1)
        criterion = torch.nn.BCELoss()
    elif TASK_TYPE == "multiclass":
        labels = torch.tensor(labels).long()
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("TASK_TYPE must be 'binary' or 'multiclass'")

    # Serverapp model
    head = None
    optimizer = None

    # Track metrics
    eval_interval = 1  # Evaluate the model every 25 rounds
    accuracies: list[tuple[int, float]] = []
    losses: list[tuple[int, float]] = []

    for i in range(1, num_rounds + 1):
        log(INFO, "")
        log(INFO, f"--- ServerApp Training Round {i} / {num_rounds} ---")

        node_ids = list(grid.get_node_ids())

        if len(node_ids) != len(in_feature_dim_clientapp):
            raise ValueError(
                "The number of feature splits must equal the number of nodes "
                f"(got {len(in_feature_dim_clientapp)} vs. {len(node_ids)})."
            )

        if head is None:
            # The server model's input size is determined by the number of clients
            # and the output feature dimension of each embedding produced by the clients
            input_dim = out_feature_dim_clientapp * len(node_ids)
            if TASK_TYPE == "binary":
                head = ServerModel(input_size=input_dim)
            else:  # multiclass
                head = ServerModel(input_size=input_dim, num_classes=OUTPUT_SIZE)
            optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

        # 1. Get embeddings from all clients
        embeddings, node_pos_mapping = get_remote_embeddings(
            grid=grid,
            node_ids=node_ids,
            num_nodes=len(labels),
            embedding_dim=out_feature_dim_clientapp,
            mode=0
        )

        # 2. Complete forward pass and compute loss
        optimizer.zero_grad()  # Clear gradients before backward pass
        output = head(embeddings)

        if TASK_TYPE == "binary":
            loss = criterion(output, labels)
        else:  # multiclass
            loss = criterion(output, labels)
        
        loss.backward()

        # 4. Extract gradients w.r.t. embeddings
        embeddings_grad = embeddings.grad.split(
            [out_feature_dim_clientapp] * len(node_ids), dim=1
        )

        # Update the head model
        optimizer.step()

        # 3. Compute accuracy using updated head model
        if i % eval_interval == 0 or i == num_rounds:
            if TASK_TYPE == "binary":
                preds = (output > 0.5).float()
                correct = (preds == labels).sum().item()
            else:
                preds = torch.argmax(output, dim=1)
                correct = (preds == labels).sum().item()
            accuracy = correct / len(labels)
            log(INFO, f"Round {i}, Loss: {loss.item():.4f}, Accuracy: {(100.0*accuracy):.2f}%")
            accuracies.append((i, accuracy))
            losses.append((i, loss.item()))

        # 5. Send gradients to clients
        send_gradients_to_clients(grid, node_pos_mapping, embeddings_grad, 0)

    # Log final results
    log(INFO, "")
    log(INFO, "=== Final Results ===")
    for (round_num, accuracy), (_, loss_value) in zip(accuracies, losses):
        log(
            INFO,
            f"Round {round_num} -> Loss: {loss_value:.4f} | Accuracy: {(100.0*accuracy):.2f}%",
        )
    
    return head


def test(dataset, num_rounds, grid, in_feature_dim_clientapp, out_feature_dim_clientapp, head):
    labels = dataset["target"]
    labels_tensor = torch.tensor(labels)
    
    # ----- TASK-SPECIFIC LABEL HANDLING -----
    if TASK_TYPE == "binary":
        labels = torch.tensor(labels).float().unsqueeze(1)
        criterion = torch.nn.BCELoss()
    elif TASK_TYPE == "multiclass":
        labels = torch.tensor(labels).long()
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("TASK_TYPE must be 'binary' or 'multiclass'")

    # Track metrics
    prediction_history = defaultdict(list)
    eval_interval = 1  # Evaluate the model every 25 rounds
    accuracies: list[tuple[int, float]] = []
    losses: list[tuple[int, float]] = []
    total_time = 0.0
    log(INFO, "")
    log(INFO, f"--- ServerApp Testing ---")

    node_ids = list(grid.get_node_ids())

    if len(node_ids) != len(in_feature_dim_clientapp):
        raise ValueError(
            "The number of feature splits must equal the number of nodes "
            f"(got {len(in_feature_dim_clientapp)} vs. {len(node_ids)})."
        )
    
    total_inference_time_ms = 0
    processed_samples = 0

    start_time = time.time()

    # 1. Get embeddings from all clients
    embeddings, _ = get_remote_embeddings(
        grid=grid,
        node_ids=node_ids,
        num_nodes=len(labels),
        embedding_dim=out_feature_dim_clientapp,
        mode=-1
    )

    # 2. Forward pass only (no gradients)
    with torch.no_grad():
        output = head(embeddings)
        if TASK_TYPE == "binary":
            loss_value = criterion(output, labels_tensor)
            predictions_probs = output.cpu().numpy().ravel()  # shape [N]
            probs = output.detach().cpu().numpy().ravel()     # shape [N]
            predictions = (predictions_probs > 0.5).astype(int)
            true_labels = labels_tensor.cpu().numpy().ravel()
        elif TASK_TYPE == "multiclass":
            loss_value = criterion(output, labels_tensor.long())
            predictions_probs = F.softmax(output, dim=1).detach().cpu().numpy()  # shape [N, C]
            probs = F.softmax(output, dim=1).detach().cpu().numpy()                         # class probabilities
            predictions = np.argmax(predictions_probs, axis=1)
            true_labels = labels_tensor.cpu().numpy()
        else:  # regression
            loss_value = None
            predictions_probs = output.cpu().numpy().ravel()
            predictions = predictions_probs.copy()
            true_labels = labels_tensor.cpu().numpy().ravel()
        processed_samples += len(labels)

    end_time = time.time()

    # Convert probabilities to binary predictions
    if TASK_TYPE == "binary":
        preds = (output >= 0.5).float()
    else:  # multiclass
        preds = torch.argmax(output, dim=1)

    # Move to CPU + numpy for sklearn
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    # Metrics
    accuracy = (y_pred == y_true).mean()
    if TASK_TYPE == "binary":
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc_micro = roc_auc_score(true_labels, probs)

        # Average inference time per sample
        total_time = end_time - start_time
        avg_inference_time = total_time / len(labels)

        # Log final results
        log(INFO, "")
        log(INFO, "=== Testing Results ===")
        log(
            INFO,
            f"Loss: {loss_value.item():.4f} | "
            f"Accuracy: {(100 * accuracy):.2f}% | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"Avg Inference Time: {avg_inference_time:.6f}s"
            f"Micro ROC-AUC: {roc_auc_micro:.6f} | "
        )

    else:  # multiclass
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        y_true_binarized = label_binarize(true_labels, classes=np.arange(OUTPUT_SIZE))
        roc_auc_macro = roc_auc_score(
            y_true_binarized,
            probs,
            multi_class="ovr",
            average="macro"
        )

        roc_auc_micro = roc_auc_score(
            y_true_binarized,
            probs,
            multi_class="ovr",
            average="micro"
        )

        # Average inference time per sample
        total_inference_time_ms += (end_time - start_time) * 1000
        avg_inference_time = total_time / len(labels)

        # Log final results
        log(INFO, "")
        log(INFO, "=== Testing Results ===")
        log(
            INFO,
            f"Loss: {loss_value.item():.4f} | "
            f"Accuracy: {(100 * accuracy):.2f}% | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"Avg Inference Time: {avg_inference_time:.6f}s"
            f"Micro ROC-AUC: {roc_auc_micro:.6f} | "
            f"Macro ROC-AUC: {roc_auc_macro:.6f} | "
        )

    # Store history
    prediction_history["predictions"] = np.array(predictions)
    prediction_history["prediction_probs"] = np.array(predictions_probs)
    prediction_history["probs"] = np.array(probs)
    prediction_history["real_values"] = np.array(true_labels)

    # Average per-sample inference time
    avg_per_sample_time_ms = total_inference_time_ms / processed_samples
    prediction_history["avg_inference_time_ms"] = np.array([avg_per_sample_time_ms])

    return prediction_history


def send_gradients_to_clients(
    grid: Grid,
    node_pos_mapping: dict[int, int],
    embeddings_grad: list[torch.Tensor],
    mode: int
) -> None:
    """Send gradients to clients."""

    # Create messages that target method in ClientApp with
    # @app.train("apply_gradients") decorator
    messages = []
    for node_id, pos in node_pos_mapping.items():
        arrc = ArrayRecord({"local-gradients": Array(embeddings_grad[pos].numpy())})
        message = Message(
            content=RecordDict({"gradients": arrc,
                                "config": ConfigRecord({
                                #"round": rnd,
                                #"batch_idx": batch_count,
                                #"pos": pos,
                                #"global_batch_size": global_batch_size,
                                #"effective_batch_size": effective_batch_size,  # <--- pass correct batch size
                                "mode": mode
                            }),}),
            message_type="train.apply_gradients",
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages, but don't wait for results (no replies expected)
    log(INFO, "Sending gradients to %s nodes...", len(messages))
    grid.push_messages(messages)


def get_remote_embeddings(
    grid: Grid,
    node_ids: list[str],
    num_nodes: int,
    embedding_dim: int,
    mode: int
) -> tuple[torch.Tensor, dict[int, int]]:
    """Get embeddings from sampled remote nodes."""

    # Create messages that target method in ClientApp with
    # @app.query("generate_embeddings") decorator
    messages = []
    for node_id in node_ids:  # one message for each node
        message = Message(
            content=RecordDict({"config": ConfigRecord({
                                #"round": rnd,
                                #"batch_idx": batch_count,
                                #"pos": pos,
                                #"global_batch_size": global_batch_size,
                                #"effective_batch_size": effective_batch_size,  # <--- pass correct batch size
                                "mode": mode
                            }),}),
            message_type="query.generate_embeddings",
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages and wait for all results
    log(INFO, "Requesting embeddings from %s nodes...", len(messages))
    replies = grid.send_and_receive(messages)
    log(INFO, "\tReceived %s/%s results", len(replies), len(messages))

    embeddings = torch.zeros((num_nodes, embedding_dim * len(node_ids)))

    # Convert all embeddings back to pytorch tensors
    # and place them in the corresponding feature segment
    node_pos_mapping: dict[int, int] = {}
    for reply in replies:
        if reply.has_content():
            arr = reply.content["arrays"]["embedding"]
            embd = torch.from_numpy(arr.numpy())
            pos = reply.content["config"]["pos"]
            node_pos_mapping[reply.metadata.src_node_id] = pos
            embeddings[:, pos * embedding_dim : (pos + 1) * embedding_dim] = embd

    return embeddings.requires_grad_(), node_pos_mapping

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
        pred_prefix = "testing"
        metric_prefix = "test"
    else:
        pred_prefix = "training"
        metric_prefix = "train"

    os.makedirs(model_directory, exist_ok=True)

    prediction_history = np.load(f"./server_model/{pred_prefix}_history_{model_name}.npz", allow_pickle=True)

    if mode == -1:
        confusion_matrix_file_name = f"{metric_prefix}_cm_{model_name}"
        y_pred = prediction_history["predictions"]
        y_proba = prediction_history["prediction_probs"]
        probs = prediction_history["probs"]
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
                y_true_bin, probs, average="micro", multi_class="ovr"
            )
        except ValueError:
            test_metrics["micro_roc_auc"] = float("nan")

        # Macro metrics
        test_metrics["macro_prec"] = precision_score(real_values, y_pred, average="macro", zero_division=0)
        test_metrics["macro_rec"] = recall_score(real_values, y_pred, average="macro", zero_division=0)
        test_metrics["macro_f1"] = f1_score(real_values, y_pred, average="macro")

        try:
            test_metrics["macro_roc_auc"] = roc_auc_score(
                y_true_bin, probs, average="macro", multi_class="ovr"
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
                    y_true_bin, probs.T, average="micro", multi_class="ovr"
                )
            except ValueError:
                test_metrics["micro_roc_auc"] = float("nan")

            # Macro metrics
            test_metrics["macro_prec"] = precision_score(real_values, y_pred, average="macro", zero_division=0)
            test_metrics["macro_rec"] = recall_score(real_values, y_pred, average="macro", zero_division=0)
            test_metrics["macro_f1"] = f1_score(real_values, y_pred, average="macro")

            try:
                test_metrics["macro_roc_auc"] = roc_auc_score(
                    y_true_bin, probs.T, average="macro", multi_class="ovr"
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