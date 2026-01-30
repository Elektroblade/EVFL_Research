"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
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
    dnn_to_arrays,
    arrays_to_dnn,
    relu
)

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = DNN(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        neurons_per_layer=NEURONS_PER_LAYER,
        learning_rate=lr,
        activation=relu,
        task_type=TASK_TYPE,
    )
    arrays = ArrayRecord(dnn_to_arrays(global_model))

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # ---- Save final model ----
    print("\nSaving final NumPy DNN model...")
    final_arrays = result.arrays
    arrays_to_dnn(global_model, final_arrays)

    with open("final_model.npy", "wb") as f:
        np.save(f, final_arrays)


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