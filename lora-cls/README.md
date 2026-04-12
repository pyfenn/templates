# LoRA Sequence Classification Template

## Overview

Fine-tunes a pre-trained transformer with LoRA (Low-Rank Adaptation) on a binary text
classification task using the fenn `LoRATrainer`. The example uses the **SST-2** sentiment
dataset (Stanford Sentiment Treebank) and **DistilBERT** as the base model.

> SST-2's public `test` split has no labels, so the HuggingFace `validation` split (872
> examples) is used as the held-out **test** set.

- **Task**: binary sentiment classification (negative / positive)
- **Dataset**: SST-2 via HuggingFace `datasets` — no manual download needed
- **Base model**: `distilbert-base-uncased` (~66 M params)
- **Trainable params**: ~740 K (~1.1 % of total) — LoRA matrices (~147 K) + randomly-initialised classifier head (~590 K)

## Architecture

LoRA inserts low-rank matrices into the query and value projections of every
DistilBERT attention layer, leaving the rest of the model frozen:

```
DistilBERT (frozen)
  └─ MultiHeadAttention
       ├─ q_lin  ←  LoRA(r=8)   ← trained
       ├─ k_lin  ←  frozen
       ├─ v_lin  ←  LoRA(r=8)   ← trained
       └─ out_lin ← frozen
  └─ classifier head             ← trained
```

**Loss**: cross-entropy (computed internally by `AutoModelForSequenceClassification`)

## Configuration

```yaml
model:
  name: distilbert-base-uncased   # any HuggingFace model ID
  max_length: 128
  num_labels: 2

lora:
  r: 8          # rank of the low-rank matrices
  alpha: 16     # scaling factor (effective lr = alpha / r)
  dropout: 0.1  # dropout on LoRA layers

train:
  seed: 42
  batch: 16
  epochs: 3
  lr: 2e-4

test:
  batch: 32
```

## Expected Results

On the SST-2 test set (872 examples) after 3 epochs over 2 000 training samples:

| Metric | Value  |
|--------|--------|
| Test Accuracy | ~ 80 % |

Training on the full 67 k split for 3 epochs reaches ~90 % accuracy.

## Customisation

**Different model** — change `model.name` and update `target_modules` in `main.py`:
- BERT: `["query", "value"]`
- RoBERTa: `["query", "value"]`
- DistilBERT: `["q_lin", "v_lin"]`

**Different dataset** — replace the `load_dataset` block in `main.py` and adjust
`model.num_labels` in `fenn.yaml`.

**More data** — remove the `[:2000]` slice on `train_texts` / `train_labels`.

## File Structure

```
lora-seq-cls/
├── main.py              # training script
├── fenn.yaml            # configuration
├── modules/
│   ├── __init__.py
│   └── dataset.py       # SentimentDataset (returns HF-style dicts)
├── requirements.txt
└── README.md
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```
