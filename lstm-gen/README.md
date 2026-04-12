# lstm-gen — Character-Level Language Model

## Overview

Trains a character-level LSTM on **TinyShakespeare** and generates new text
after training. The dataset (~1 MB) is downloaded automatically from
HuggingFace Hub on first run.

- **Task**: Next-character prediction (autoregressive sequence generation)
- **Dataset**: TinyShakespeare (`Trelis/tiny-shakespeare`, ~1 M chars)
- **Model**: Embedding + stacked LSTM + linear head

## How it works

The corpus is tokenised at the **character level** (~65 unique chars for
Shakespeare). Training creates sliding-window pairs:

```
input : T H E   K I N G
target: H E   K I N G ' s
```

The model learns to predict the next character at every position, minimising
cross-entropy loss. After training, text is generated autoregressively from a
seed prompt using temperature sampling.

## Configuration

```yaml
data:
  seq_len: 128        # context window (chars)

model:
  embed_dim: 64       # character embedding size
  hidden_dim: 256     # LSTM hidden units
  num_layers: 2
  dropout: 0.3

generate:
  prompt: "ROMEO:"
  max_new_tokens: 500
  temperature: 0.8    # lower = safer, higher = wilder
```

## Temperature

| Temperature | Effect |
|-------------|--------|
| `0.5` | Repetitive but coherent |
| `0.8` | Good balance (default) |
| `1.0` | Unscaled, diverse |
| `1.2+` | Creative / noisy |

## Expected Results

After 10 epochs on a GPU:

| Metric | Value |
|--------|-------|
| Val loss | ~1.4–1.6 |

The generated text will resemble Shakespearean syntax without being
grammatically perfect — this is expected for a character-level model without
an attention mechanism.

## Swapping the corpus

Replace the `hf_hub_download` call with any plain-text file:

```python
text = open("my_corpus.txt", encoding="utf-8").read()
```

The vocabulary and dataset are built automatically from whatever text you
provide. No other changes needed.

## File Structure

```
lstm-gen/
├── main.py          # training + generation script
├── fenn.yaml        # configuration
├── requirements.txt
├── README.md
└── modules/
    ├── dataset.py   # CharDataset — sliding-window char sequences
    └── model.py     # CharLM — Embedding + LSTM + linear head + generate()
```

## Running

```bash
pip install -r requirements.txt
python main.py
```
