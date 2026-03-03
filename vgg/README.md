# VGG16 Image Classification (template)

This template demonstrates how to train a **VGG16 convolutional neural network** on an image dataset using the FENN framework.

By default it uses **CIFAR-10** as an example, showing the full pipeline of
- data augmentation and preprocessing
- model creation and training with a generic `Trainer`
- evaluation on a held‑out test set

The network architecture is a faithful implementation of the classic VGG16 model,
adapted to handle arbitrary numbers of output classes via a configurable `num_classes`.

## Architecture

The `VGG16` class defined in `modules/model.py` consists of:

- 13 convolutional layers grouped in blocks, each with BatchNorm and ReLU
- Max‑pooling layers after blocks 2, 4, 7, 10 and 13 to downsample spatially
- Three fully‑connected layers (4096 → 4096 → `num_classes`), with Dropout

For 224×224 inputs the spatial size after the final pooling is 7×7, yielding
`7*7*512` features before the dense layers.

## Configuration

Hyperparameters and dataset options are defined in `fenn.yaml`:

```yaml
project: vgg

logger:
  dir: logger

general:
  device: cuda

model:
  in_channels: 3
  num_classes: 10

train:
  seed: 42
  batch: 32
  epochs: 1
  lr: 1e-3

test:
  size: 0.2
  batch: 32

dataset:
  dir: data
```

### Key parameters

| Key | Default | Description |
|-----|---------|-------------|
| `general.device` | cuda | compute device (cpu or cuda) |
| `model.num_classes` | 10 | number of target classes |
| `train.seed` | 42 | random seed for reproducibility |
| `train.batch` | 32 | training batch size |
| `train.epochs` | 1 | number of training epochs |
| `train.lr` | 1e-3 | learning rate for Adam |
| `test.size` | 0.2 | fraction of data held out for test (unused for CIFAR) |
| `test.batch` | 32 | batch size for evaluation |
| `dataset.dir` | data | directory where CIFAR datasets are downloaded |

## Data

The example pipeline downloads CIFAR‑10 and applies the following transforms:

- random horizontal flip, random 32×32 crop with padding, resize to 224×224
- normalization to `mean=(0.5,0.5,0.5)` and `std=(0.5,0.5,0.5)`

Replace the dataset loading section in `main.py` with your own dataset as needed.

## Running the template

Install requirements and run:

```bash
cd vgg
pip install -r requirements.txt
python main.py
```

You can override configuration via environment variable:

```bash
FENN_CONFIG=custom.yaml python main.py
```

The script will print training progress (managed by FENN) and output a final
accuracy score on the test set, for example:

```
Accuracy: 0.7345
```

## Outputs

This template does not write any checkpoints by default, but the FENN logger will
record training metrics if enabled. Use the printed accuracy to gauge performance.

## Notes

- The VGG16 model is relatively large; training on CIFAR‑10 with default settings
  may be slow on CPU. Use a GPU for reasonable speed.
- To fine‑tune on a different dataset, adjust `num_classes`, data transforms, and
  optionally resize inputs appropriately.

---

This README provides a quick starting point for experimenting with convolutional
classification tasks using the VGG architecture within the FENN framework.