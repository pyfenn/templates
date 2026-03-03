# Multiclass MLP Template

## Overview

The Multiclass MLP (Multi-Layer Perceptron) template is designed for **multiclass classification tasks** with more than two classes. It demonstrates how to:
- Build a neural network for multiclass classification on tabular data
- Handle multiple output classes with CrossEntropyLoss
- Train using the FENN framework with validation and early stopping
- Evaluate model performance on multiclass classification metrics

This template uses the Iris dataset as an example, classifying flowers into three species (Setosa, Versicolor, Virginica).

## Architecture

### MultiClassMLP Model

A 3-layer feedforward neural network optimized for multiclass classification:

```
Input (4 features)
    ↓
Linear(4 → 18) → ReLU
    ↓
Linear(18 → 16) → ReLU
    ↓
Linear(16 → 3)  [for 3 classes]
    ↓
Output (3 logits for multiclass classification)
```

**Architecture Details:**
- **Layer 1**: 4 input features → 18 hidden units with ReLU activation
- **Layer 2**: 18 → 16 hidden units with ReLU activation
- **Output Layer**: 16 → number of classes (logits)

**Loss Function**: CrossEntropyLoss (combines softmax and negative log-likelihood)

### Dataset

The custom `MultiClassDataset` class:
- Takes numpy arrays (features X and labels y)
- Returns torch tensors for training
- Automatically converts labels to torch.long format required by CrossEntropyLoss
- Squeezes label dimensions properly for multiclass training

## Configuration

The template uses YAML configuration for hyperparameters:

### Example fenn.yaml

```yaml
project: mlp

logger:
  dir: logger

general:
  device: cuda

train:
  seed: 42
  batch: 16
  epochs: 30
  lr: 4e-3

test:
  size: 0.2
  batch: 16
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `general.device` | str | Device to use (cpu, cuda) | cuda |
| `train.seed` | int | Random seed for reproducibility | 42 |
| `train.batch` | int | Batch size for training | 16 |
| `train.epochs` | int | Maximum number of training epochs | 30 |
| `train.lr` | float | Learning rate for optimizer | 0.004 |
| `test.size` | float | Proportion of data for testing (0-1) | 0.2 |
| `test.batch` | int | Batch size for testing | 16 |


## Key Features

### Multiclass Classification
- Uses CrossEntropyLoss which combines softmax and negative log-likelihood
- Output has `num_classes` logits (3 for Iris dataset)
- Predictions use argmax to select the class with highest probability
- Supports any number of classes (2 or more)

### Validation & Early Stopping
- Validates every 5 epochs on the test set during training
- Automatically stops after 5 epochs without improvement (early stopping)
- Saves the best model to `./checkpoints/checkpoint_best.pt`
- During validation, computes: accuracy, precision, recall, F1-score

**Validation Process:**
```python
# Train with validation on test set (validation runs every epoch)
model = trainer.fit(
  train_loader=train_loader,
  val_loader=test_loader,  # Validation set
)
```

The trainer automatically monitors validation loss to:
1. Determine the best model (saves when val_loss improves)
2. Trigger early stopping (stops if val_loss doesn't improve for 5 epochs)
3. Report validation metrics (accuracy, precision, recall, F1)

### Data Preprocessing
- **Feature Scaling**: StandardScaler normalizes input features
- **Train-Test Split**: 80% training, 20% testing with stratification
- **Custom Dataset**: MultiClassDataset handles tensor conversion
- **Label Mapping**: String labels are converted to integer class indices

### Reproducibility
- Fixed random seed ensures consistent results
- Stratified split maintains class distribution across splits

## Customization

### Using Different Data with Custom Number of Classes

To use your own dataset with different number of classes:

```python
# Load your own CSV
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1).values
y = df["target"].values.astype(np.int64)

# No need to modify num_classes - it's computed automatically
num_classes = np.unique(y).shape[0]
```

### Adjusting Model Architecture for Different Input Sizes

To modify the MLP for different input dimensions:

```python
class CustomMultiClassMLP(nn.Module):
    def __init__(self, input_size=4, num_classes=3, hidden_sizes=[32, 16]):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Usage with different dimensions
model = CustomMultiClassMLP(input_size=10, num_classes=5, hidden_sizes=[64, 32])
```

### Adjusting Hyperparameters

Modify `fenn.yaml`:

```yaml
train:
  seed: 42
  batch: 32        # Increase batch size
  epochs: 50       # Train longer
  lr: 2e-3         # Lower learning rate
```

### Customizing Early Stopping

Adjust patience or validation frequency:

```python
trainer = Trainer(
  ...,
  early_stopping_patience=10,  # Wait 10 epochs without improvement
)

model = trainer.fit(
  train_loader=train_loader,
  val_loader=test_loader,  # validation will be checked every epoch
)
```

## Performance Monitoring

The trainer automatically logs during training:
- Training loss per epoch
- Validation accuracy, precision, recall, F1-score
- Early stopping events
- Best model checkpoint location

Example output:
```
Epoch 0. Mean Loss: 1.0986
Epoch 5. Validation Loss: 0.8234
Epoch 5. Validation Accuracy: 0.9667
Epoch 5. Validation Precision: 0.9667
Epoch 5. Validation Recall: 0.9667
Epoch 5. Validation F1 Score: 0.9667
...
Early stopping triggered. No improvement for 5 epochs.
Best model checkpoint saved to ./checkpoints/checkpoint_best.pt
```

## Expected Results

On the Iris multiclass classification task (3 species):
- Training accuracy: ~97-99%
- Test accuracy: ~93-97%

Results depend on:
- Random seed
- Train-test split
- Hyperparameters
- Model architecture

## File Structure

```
mlp-multiclass/
├── main.py                       # Training script
├── modules/
│   ├── __init__.py              # Module imports
│   ├── model.py                 # MultiClassMLP architecture
│   └── dataset.py               # MultiClassDataset class
├── fenn.yaml                    # Configuration file
├── requirements.txt             # Python dependencies
└── mlp_multiclass.md           # This documentation
```

## Dependencies

Required packages:
- PyTorch
- scikit-learn (for preprocessing and metrics)
- numpy
- pandas
- ucimlrepo (for fetching UCI datasets)
- FENN framework

Install with:
```bash
pip install torch scikit-learn numpy pandas ucimlrepo
```

## Running the Template

```bash
# Using FENN CLI
python main.py

# With custom config
FENN_CONFIG=custom.yaml python main.py

# Check available devices
python -c "import torch; print(torch.cuda.is_available())"
```

## Multiclass Classification Metrics

The trainer automatically computes during validation:

- **Accuracy**: Proportion of correct predictions
- **Precision**: Weighted average across all classes
- **Recall**: Weighted average across all classes
- **F1 Score**: Harmonic mean of precision and recall

For multiclass problems with imbalanced datasets, consider using:
- **Macro averaging**: Equal weight to each class
- **Weighted averaging**: Weight by class frequency (default)

## Comparing with Binary Classification

| Aspect | Binary (BinaryMLP) | Multiclass (MultiClassMLP) |
|--------|-------------------|----------------------------|
| Output Units | 1 logit | num_classes logits |
| Loss Function | BCEWithLogitsLoss | CrossEntropyLoss |
| Prediction | Sigmoid + threshold | Softmax + argmax |
| Label Format | torch.float | torch.long |
| Classes | 2 | 3+ |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch` size in fenn.yaml |
| Slow training | Enable GPU: set `device: cuda` |
| Poor accuracy | Try lower learning rate (2e-3) or more epochs |
| Overfitting | Add regularization or use smaller hidden layers |
| NaN loss | Check data normalization and learning rate |
| Class imbalance issues | Use weighted sampler or class weights in loss |

## Extending to Larger Datasets

For production use with larger datasets:

1. **Avoid loading entire dataset into memory**:
```python
# Use generators or streaming data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4  # Parallel data loading
)
```

2. **Add data validation checks**:
```python
assert X_train.shape[1] == 4, "Expected 4 features"
assert num_classes == 3, "Expected 3 classes"
```

3. **Monitor GPU memory**:
```bash
# Check GPU usage
nvidia-smi
```

## References

- [PyTorch Multiclass Classification](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Softmax and CrossEntropy Loss](https://towardsdatascience.com/softmax-and-cross-entropy-loss-ae5e58670e60)
- FENN Framework Documentation
