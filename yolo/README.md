# YOLO Object Detection Template

## Overview

The YOLO Object Detection template is designed for **computer vision tasks** requiring bounding box localization and object classification. It demonstrates how to:
- Integrate the powerful `ultralytics` YOLO engine seamlessly into the PyFenn framework
- Fine-tune pre-trained models (Transfer Learning) on custom datasets
- Handle PyFenn configuration (`fenn.yaml`) to control third-party backends
- Enforce clean directory structures (`logger/` and `models/`) by gracefully wrapping heavy, framework-specific outputs

This template uses the lightweight `coco8` dataset as an example, automatically downloading 8 sample images and their bounding box labels to verify your training pipeline instantly.

## Architecture

### YOLOv8 Model

The template defaults to the YOLOv8 Nano architecture, optimized for real-time object detection:

```text
Input Image (e.g., 640x640)
    ↓
Backbone (CSPDarknet - Feature Extraction)
    ↓
Neck (PANet - Feature Fusion across scales)
    ↓
Head (Decoupled Head)
    ↓
Output (Bounding Box Coordinates [x, y, w, h] + Class Probabilities)
```

**Architecture Details:**
- **Weights**: Uses `yolov8n.pt` pre-trained on the COCO dataset (transfer learning).
- **Loss Functions**: A combination of Box Loss (CIoU), Class Loss (Cross Entropy), and DFL (Distribution Focal Loss) handled internally by the engine.

### Dataset Handling

Unlike pure PyTorch templates that require custom `modules/dataset.py` classes, this template leverages the native YOLO format.
- Takes a `.yaml` file pointing to `images/` and `labels/` directories.
- The engine automatically handles data loading, caching, resizing (`imgsz`), and complex augmentations (Mosaic, MixUp) under the hood.

## Configuration

The template uses YAML configuration for hyperparameters and backend control:

### Example fenn.yaml

```yaml
project: yolo-object-detection

logger:
  dir: logger

general:
  device: cuda

model:
  weights: yolov8n.pt

data:
  dataset_yaml: coco8.yaml

train:
  seed: 42
  batch: 16
  epochs: 50
  imgsz: 640
  verbose: False

checkpoint:
  save_dir: ./models
  epochs: 5
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `general.device` | str | Device to use (cpu, cuda) | cuda |
| `model.weights` | str | Pre-trained weights to load | yolov8n.pt |
| `data.dataset_yaml` | str | Path to the dataset configuration file | coco8.yaml |
| `train.seed` | int | Random seed for reproducibility | 42 |
| `train.batch` | int | Batch size for training | 16 |
| `train.epochs` | int | Maximum number of training epochs | 50 |
| `train.imgsz` | int | Image size to resize to before training | 640 |
| `train.verbose` | bool | Suppress heavy terminal outputs | False |
| `checkpoint.save_dir`| str | Directory to save the final `best_yolo_model.pt` | ./models |
| `checkpoint.epochs` | int | Frequency to save intermediate weights | 5 |

## Key Features

### PyFenn Structure Compliance (The "Ghost Directory")
Third-party engines like Ultralytics typically generate heavy, hardcoded `runs/` folders containing hundreds of charts and intermediate files. This template uses a Python `tempfile.TemporaryDirectory()` to sandbox the training engine:
1. Forces YOLO to write all intermediate files to a temporary OS folder.
2. Extracts *only* the `best.pt` weights and moves them cleanly to PyFenn's `models/` directory.
3. Automatically deletes the temporary folder, keeping your workspace perfectly clean while maintaining standard `.log` outputs in FENN's `logger/`.

### Transfer Learning
- Automatically downloads base weights (e.g., `yolov8n.pt`).
- Freezes appropriate base layers and fine-tunes the detection head for your specific classes, reducing training time from weeks to minutes.

### Auto-Optimization
- Automatically inspects the dataset size and adjusts the optimizer (e.g., switching to AdamW with aggressive learning rate decay for very small datasets).

## Customization

### Using Custom Data

To train on your own images, you must provide a YOLO dataset configuration YAML.

1. **Organize your dataset** into folders:
```text
custom_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

2. **Create a `custom.yaml` file:**
```yaml
path: /absolute/path/to/custom_dataset
train: images/train
val: images/val

nc: 2
names: ['car', 'pedestrian']
```

3. **Update `fenn.yaml`:**
```yaml
data:
  dataset_yaml: custom.yaml
```

### Adjusting Model Size

If you need higher accuracy and have the GPU memory, simply change the weights in `fenn.yaml`:

```yaml
model:
  weights: yolov8s.pt  # Upgrades from Nano to Small
```

## Performance Monitoring

The script evaluates the model upon completion and prints the standard object detection metric.

Example output:
```text
[INFO] Starting training phase on dataset: coco8.yaml
...
[INFO] Saved model weights to ./models/best_yolo_model.pt
mAP50-95: 0.6421
```

## Expected Results

On the COCO8 toy dataset (50 epochs, Nano model):
- **mAP50-95**: ~0.60 - 0.65
- **mAP50**: ~0.85 - 0.90
- **Inference Speed**: ~1-2ms per image (GPU) / ~70-80ms per image (CPU)

*Note: COCO8 is an extremely small dataset meant strictly for pipeline verification. Real-world datasets require hundreds or thousands of images to achieve high mAP.*

## File Structure

Because the Ultralytics engine handles the PyTorch `nn.Module` architecture and custom DataLoaders internally, this template does not require a `modules/` folder.

```text
yolo-object-detection/
├── main.py                       # Training script and engine wrapper
├── fenn.yaml                     # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation
```
*(The `logger/` and `models/` directories are generated automatically during execution).*

## Dependencies

Required packages:
- PyTorch
- ultralytics (>=8.0.0)
- FENN framework

Install with:
```bash
pip install -r requirements.txt
```

## Running the Template

```bash
# Standard execution
python main.py
```

## Object Detection Metrics

Unlike standard classification (Accuracy/Precision), object detection uses bounding box metrics:

- **IoU (Intersection over Union)**: Measures the overlap between the predicted bounding box and the ground truth box.
- **mAP50**: Mean Average Precision calculated at an IoU threshold of 0.50.
- **mAP50-95**: The primary metric. Averages the mAP over multiple IoU thresholds (from 0.50 to 0.95 in steps of 0.05). It strictly rewards models that draw highly precise, tight bounding boxes.

## Comparing YOLO Models

| Model Size | Weights Parameter | Speed (Relative) | Accuracy (Relative) | Best For |
|------------|-------------------|------------------|---------------------|----------|
| Nano | `yolov8n.pt` | Fastest | Baseline | Edge devices, Raspberry Pi |
| Small | `yolov8s.pt` | Fast | Good | Standard real-time web tracking |
| Medium | `yolov8m.pt` | Moderate | High | Complex scenes, server-side inference |
| Large | `yolov8l.pt` | Slow | Highest | High-end GPU batch processing |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `train.batch` and/or `train.imgsz` in `fenn.yaml`. |
| `Images not found` error | Check that the `path` variable in your dataset YAML uses an **absolute path**, or that your relative paths perfectly match your execution directory. |
| Zero or NaN mAP | Check your `.txt` label files. They must be normalized (values between 0 and 1) and formatted as: `class_id center_x center_y width height`. |
| No bounding boxes detected | Ensure `nc` (number of classes) in your data YAML strictly matches the highest `class_id` in your labels. |

## Extending to Larger Datasets

For production models:
1. **Enable advanced augmentations**: Add YOLO augmentation parameters directly to the `train:` block in `fenn.yaml` (e.g., `mosaic: 1.0`, `mixup: 0.1`).
2. **Implement Early Stopping**: Add `patience: 50` to the `train:` block to halt training if metrics degrade.

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Understanding mAP in Object Detection](https://blog.roboflow.com/mean-average-precision/)
- FENN Framework Documentation