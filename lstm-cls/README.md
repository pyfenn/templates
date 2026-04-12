# LSTM Sequence Classification Template

## Overview

Trains a bidirectional LSTM on the **WISDM Human Activity Recognition** dataset using
the fenn `ClassificationTrainer`. Data is streamed from HuggingFace — no manual
download needed.

- **Task**: 6-class activity recognition from accelerometer sequences
- **Dataset**: WISDM via HuggingFace `datasets`
- **Model**: bidirectional LSTM with configurable pooling + MLP head

## Dataset

[monster-monash/WISDM](https://huggingface.co/datasets/monster-monash/WISDM) — 17,166
3-axis accelerometer sequences (100 timesteps = 5 s at 20 Hz) collected from 36
participants carrying a smartphone.

| Class | Activity     |
|-------|--------------|
| 0     | Walking      |
| 1     | Jogging      |
| 2     | Stairs       |
| 3     | Sitting      |
| 4     | Standing     |
| 5     | Lying Down   |

> The dataset bundles a custom loading script which is no longer supported by
> recent versions of the `datasets` library. `main.py` fetches the raw
> `WISDM_X.npy` / `WISDM_y.npy` files directly via `huggingface_hub` and
> splits them with sklearn. Files are cached locally after the first run.
>
> **Raw format**: `X` is stored as `(N, 3, 100)` (channels-first).
> `main.py` transposes it to `(N, 100, 3)` before passing it to the LSTM.

## Architecture

```
Input (B, T=100, F=3)
  └─ BiLSTM (hidden=128, layers=2)
       └─ Last-step pooling → (B, 256)
            └─ LayerNorm → Linear → ReLU → Dropout → Linear(6)
```

**Loss**: cross-entropy

## Configuration

```yaml
data:
  fold: fold_0   # fold_0 … fold_4

model:
  hidden_dim: 128
  num_layers: 2
  num_classes: 6
  dropout: 0.3
  bidirectional: true
  pooling: last   # last | mean | max

train:
  seed: 42
  batch: 64
  epochs: 20
  lr: 1e-3
```

## Expected Results

After 20 epochs on fold_0:

| Metric        | Value   |
|---------------|---------|
| Test Accuracy | ~90–95% |

## Customisation

**Real data** — replace the `hf_hub_download` block in `main.py` with your own NumPy
arrays and keep the `SeqDataset` wiring as-is.

**Different sequence length / features** — update `data.seq_len` and `data.features`
in `fenn.yaml` to match your data, and adjust `input_dim` in the model init.

**Unidirectional LSTM** — set `model.bidirectional: false` in `fenn.yaml`.

## File Structure

```
lstm-dev-only/
├── main.py              # training script
├── fenn.yaml            # configuration
├── modules/
│   ├── __init__.py
│   ├── dataset.py       # SeqDataset
│   └── model.py         # LSTMClassifier
├── requirements.txt
└── README.md
```

## Running

```bash
pip install -r requirements.txt
python main.py
```
