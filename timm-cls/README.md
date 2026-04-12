# timm Image Classification Template

## Overview

Fine-tunes any [timm](https://github.com/huggingface/pytorch-image-models) backbone on
**CIFAR-10** using the fenn `ClassificationTrainer`. The dataset is downloaded
automatically on first run; no manual setup needed.

- **Task**: 10-class image classification
- **Dataset**: CIFAR-10 via torchvision (auto-download)
- **Model**: any timm backbone with the classifier head replaced for `num_classes`

## Fine-tuning modes

Controlled by `model.freeze_backbone` in `fenn.yaml`:

| Mode | Setting | What trains |
|------|---------|-------------|
| Head-only | `freeze_backbone: true` | classifier head only (~1–2 M params) |
| Full fine-tuning | `freeze_backbone: false` | all weights |

Head-only is faster and less prone to overfitting on small datasets.
Switch to full fine-tuning when you have more data or want maximum accuracy.

## Configuration

```yaml
model:
  name: resnet50       # swap for any timm model
  pretrained: true
  freeze_backbone: true
  num_classes: 10

train:
  seed: 42
  batch: 64
  epochs: 5
  lr: 1e-3
  weight_decay: 1e-4
```

## Expected Results

Head-only fine-tuning of `resnet50` on CIFAR-10, 5 epochs:

| Metric        | Value   |
|---------------|---------|
| Test Accuracy | ~80–85% |

Full fine-tuning for 10+ epochs typically exceeds 90%.

## Swapping the backbone

Change `model.name` in `fenn.yaml` to any timm model ID. Some fast options:

```yaml
model.name: resnet18            # lighter ResNet
model.name: efficientnet_b0     # efficient family
model.name: mobilenetv3_small_100
model.name: convnext_tiny
```

For Vision Transformers (larger GPU memory required):

```yaml
model.name: vit_small_patch16_224
model.name: vit_base_patch16_224
```

Browse all models: `python -c "import timm; print(timm.list_models())"`.

## Customisation

**Different dataset** — replace `datasets.CIFAR10` with `datasets.ImageFolder` or
any torchvision dataset. Update `model.num_classes` to match.

**Different image size** — timm models expose their native resolution via
`timm.get_pretrained_cfg(name).input_size`. Update the `Resize` / `CenterCrop`
values in `main.py` accordingly.

## File Structure

```
timm-cls/
├── main.py          # training script
├── fenn.yaml        # configuration
├── requirements.txt
└── README.md
```

## Running

```bash
pip install -r requirements.txt
python main.py
```
