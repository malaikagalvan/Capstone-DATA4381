# ============================================================
# SAVE FINAL NEURAL NETWORK BASELINE
# ============================================================

from pathlib import Path
import joblib
import numpy as np


# ------------------------------------------------------------
# Save location
# ------------------------------------------------------------

MODEL_DIR = Path("../saved_models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR
    / "nn_baseline.joblib"
)


# ------------------------------------------------------------
# Package everything needed to reproduce / inspect the model
# ------------------------------------------------------------

saved_model = {
    # Trained neural network
    "model": model,

    # Class information
    "class_names": list(CLASS_NAMES),

    # Architecture information
    "architecture": {
        "input_shape": (48, 48),
        "flattened_size": 2304,
        "hidden_size": 128,
        "output_size": 4,
        "layers": [
            "Flatten",
            "Dense(2304, 128)",
            "ReLU",
            "Dense(128, 4)",
        ],
    },

    # Training configuration
    "training_config": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "train_ratio": 0.80,
        "test_ratio": 0.20,
    },

    # Final metrics
    "metrics": {
        "final_train_loss": train_history[-1]["loss"],
        "final_train_accuracy": train_history[-1]["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": overall_accuracy,
        "balanced_accuracy": balanced_accuracy,
    },

    # Optional training history
    "train_history": train_history,
}


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

joblib.dump(
    saved_model,
    MODEL_PATH,
)


print(
    "Neural network baseline saved successfully."
)

print(
    f"Saved to: {MODEL_PATH.resolve()}"
)
