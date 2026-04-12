# LeNet Template

## Overview

LeNet is a simple Convolutional Neural Network (CNN) template designed for image classification tasks. The architecture is inspired by the classic LeNet-5 design and is optimized for small to medium-sized images (32x32 RGB images like CIFAR-10).

This template demonstrates how to:
- Define a CNN architecture with batch normalization and dropout
- Set up training with the FENN framework
- Evaluate model performance
- Use the FENN Trainer class for training management

## Architecture

### LeNet Model

The LeNet model consists of:

1. **Three Convolutional Blocks**, each containing:
   - Convolutional layer (Conv2d)
   - Batch Normalization
   - ReLU activation
   - Max Pooling

2. **Fully Connected Layers**:
   - Hidden layer with 256 units and dropout
   - Output layer with `num_classes` units

**Architecture Details:**

```
Input (3, 32, 32)
    ↓
Conv2d(3→32, 3x3, padding=1) → BatchNorm → ReLU → MaxPool(2x2) [32→16]
    ↓
Conv2d(32→64, 3x3, padding=1) → BatchNorm → ReLU → MaxPool(2x2) [16→8]
    ↓
Conv2d(64→128, 3x3, padding=1) → BatchNorm → ReLU → MaxPool(2x2) [8→4]
    ↓
Flatten (128*4*4 = 2048)
    ↓
Linear(2048→256) → ReLU → Dropout(0.25)
    ↓
Linear(256→num_classes)
    ↓
Output (num_classes)
```

### Model Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `in_channels` | int | Number of input channels (3 for RGB) | 3 |
| `num_classes` | int | Number of output classes | 10 |

## Configuration

The LeNet template uses a YAML configuration file (`fenn.yaml`) to manage hyperparameters and settings.

### Example Configuration

```yaml
project: lenet

logger:
  dir: logger

general:
  device: cuda

model:
  in_channels: 3
  num_classes: 10

train:
  seed: 42
  batch: 128
  epochs: 50
  lr: 0.001

test:
  batch: 128

dataset:
  dir: data/
```

## Features Included

The LeNet template now includes:

- **Early Stopping**: Automatically stops training after 5 epochs without improvement
- **Model Checkpointing**: Saves the best model during training
- **Validation During Training**: Validates every 5 epochs to monitor performance
- **Data Augmentation**: Random horizontal flips and crops for robustness

## Customizing Early Stopping and Validation

You can modify the trainer parameters in the basic training code:

```python
# For more patience before stopping (wait 10 epochs without improvement)
trainer = Trainer(
    ...,
    early_stopping_patience=10
)

# Validation is performed every epoch when a validation loader is provided
trained_model = trainer.fit(
  train_loader=train_loader,
  val_loader=test_loader,
)

# To disable early stopping
trainer = Trainer(
    ...,
    early_stopping_patience=None  # No early stopping
)
```

## Customization

### Modifying the Model Architecture

To adapt the LeNet model for different image sizes or configurations:

```python
from modules import LeNet

# For smaller images (28x28, like MNIST)
model = LeNet(in_channels=1, num_classes=10)

# For larger images (224x224), you would need to adjust layer sizes
# Example: adjust fc1 size based on final pool output dimensions
```

### Using Different Datasets

To use a different dataset instead of CIFAR-10:

```python
# Example: Using MNIST
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(
    root=args["dataset"]["dir"],
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

# Adjust model for MNIST (1 channel, 28x28 images)
model = LeNet(in_channels=1, num_classes=10)
```

### Adjusting Hyperparameters

Modify `fenn.yaml` to experiment with different settings:

```yaml
train:
  seed: 42
  batch: 64      # Reduce batch size
  epochs: 100    # Increase epochs
  lr: 0.0005     # Lower learning rate
```

## Training Performance Tips

1. **Batch Normalization**: Helps stabilize training and allows higher learning rates
2. **Dropout**: Prevents overfitting (currently set to 0.25)
3. **Data Augmentation**: Random horizontal flip and crops improve generalization
4. **Learning Rate**: Start with 0.001 and adjust based on convergence
5. **Seed**: Set seed for reproducible results

## Expected Results

When training on CIFAR-10 for 50 epochs:
- Training accuracy: ~95%
- Test accuracy: ~85-90%

Results may vary based on:
- Random seed
- Exact hyperparameters
- Hardware (GPU optimization may differ)

## File Structure

```
lenet/
├── main.py                 # Training script
├── modules/
│   ├── __init__.py        # Module imports
│   └── model.py           # LeNet architecture
├── fenn.yaml              # Configuration file
└── requirements.txt       # Python dependencies
```

## Dependencies

The template requires:
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- FENN framework

Install with:
```bash
pip install torch torchvision numpy pandas scikit-learn
```

## Running the Template

```bash
# Using FENN CLI
python main.py

# Or with custom configuration
FENN_CONFIG=custom.yaml python main.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size in `fenn.yaml` |
| Slow training | Enable GPU in `fenn.yaml` (set `device: cuda`) |
| Model not converging | Try lower learning rate (e.g., 0.0005) |
| Poor accuracy | Increase number of epochs or use data augmentation |

## References

- [LeNet-5 Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- FENN Framework Documentation