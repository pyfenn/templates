from fenn import Fenn
from fenn.utils import set_seed
from ultralytics import YOLO

import torch
import os
import shutil
import tempfile

app = Fenn()


@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"])  # set seed for reproducibility

    # ========================================
    # TODO: REPLACE WITH YOUR ACTUAL DATA YAML
    # ========================================
    dataset = args["data"]["dataset_yaml"]
    weights = args["model"]["weights"]
    # ========================================

    model = YOLO(weights)

    save_period = args.get("checkpoint", {}).get("epochs", -1)

    # Use a temporary directory to trap Ultralytics' default file hoarding
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Train Phase
        model.train(
            data=dataset,
            device=device,
            project=temp_dir,  # Force YOLO to build its runs/ folder in the temp dir
            name="train_run",
            save_period=save_period,
            plots=False,  # Suppress chart generation
            **args["train"],
        )

        # 2. Extract the best model to the PyFenn designated checkpoint directory
        if "checkpoint" in args and "save_dir" in args["checkpoint"]:
            save_dir = args["checkpoint"]["save_dir"]
            os.makedirs(save_dir, exist_ok=True)

            best_weights = os.path.join(temp_dir, "train_run", "weights", "best.pt")
            if os.path.exists(best_weights):
                final_model_path = os.path.join(save_dir, "best_yolo_model.pt")
                shutil.copy(best_weights, final_model_path)
                print(f"[INFO] Saved model weights to {final_model_path}")

        # 3. Test / Evaluate Phase
        metrics = model.val(project=temp_dir, name="val_run", plots=False)

    # The 'with' block ends here, automatically deleting the temp_dir and all YOLO junk files

    # 4. Final Metric Output
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    app.run()
