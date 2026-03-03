# Regression MLP Template

## Overview

The Regression MLP (Multi-Layer Perceptron) template is designed for **regression tasks** where the goal is to predict continuous numerical values. It demonstrates how to:
- Build a neural network for regression on tabular data
- Handle continuous output values with MSELoss
- Train using the FENN framework with validation and early stopping
- Evaluate model performance using regression metrics

This template uses the Housing Price Prediction dataset as an example, predicting house prices based on property features.

## Architecture

### RegressionMLP Model

A 3-layer feedforward neural network optimized for regression:

```
Input (features)
    ↓
Linear(input_size → 32) → ReLU
    ↓
Linear(32 → 16) → ReLU
    ↓
Linear(16 → 1)  [single continuous output]
    ↓
Output (single scalar value for prediction)
```

**Architecture Details:**
- **Layer 1**: Input features → 32 hidden units with ReLU activation
- **Layer 2**: 32 → 16 hidden units with ReLU activation
- **Output Layer**: 16 → 1 (single continuous value for regression)

**Loss Function**: MSELoss (Mean Squared Error) - measures average squared difference between predictions and targets

### Dataset

The custom `RegressionDataset` class:
- Takes numpy arrays (features X and continuous targets y)
- Returns torch tensors for training
- Automatically converts both features and targets to torch.float format
- Handles different input dimensions automatically

## Configuration

The template uses YAML configuration for hyperparameters:

### Example fenn.yaml

```yaml
project: mlp

logger:
  dir: logger

general:
  device: cuda
  dataset_dir: ./data

train:
  seed: 42
  batch: 16
  epochs: 50
  lr: 1e-3

test:
  size: 0.2
  batch: 16
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `general.device` | str | Device to use (cpu, cuda) | cuda |
| `general.dataset_dir` | str | Directory for downloaded datasets | ./data |
| `train.seed` | int | Random seed for reproducibility | 42 |
| `train.batch` | int | Batch size for training | 16 |
| `train.epochs` | int | Maximum number of training epochs | 50 |
| `train.lr` | float | Learning rate for optimizer | 0.001 |
| `test.size` | float | Proportion of data for testing (0-1) | 0.2 |
| `test.batch` | int | Batch size for testing | 16 |

## Key Features

### Regression Task
- Predicts continuous numerical values (e.g., house prices)
- Uses MSELoss which penalizes larger errors more heavily than smaller ones
- Single output neuron returns continuous value without activation function
- Supports any numerical prediction problem

### Validation & Early Stopping
- Validates every epoch on the validation set during training
- Automatically stops after 5 epochs without improvement (early stopping)
- Saves the best model to `./checkpoints/checkpoint_best.pt`
- During validation, computes: MSE, MAE, R² score

**Validation Process:**
```python
# Train with validation on separate set
model = trainer.fit(
  train_loader=train_loader,
  val_loader=val_loader,  # Validation set
)
```

The trainer automatically monitors validation loss to:
1. Determine the best model (saves when val_loss improves)
2. Trigger early stopping (stops if val_loss doesn't improve for 5 epochs)
3. Report validation metrics

### Data Preprocessing
- **Feature Scaling**: StandardScaler normalizes input features to zero mean and unit variance
- **Target Scaling**: StandardScaler also normalizes target values for better training stability
- **Categorical Encoding**: One-hot encoding converts categorical features to numerical format
- **Train-Validation-Test Split**: 70% training, 15% validation, 15% testing with random split

### Normalization Best Practice
- **Fit only on training data**: StandardScaler.fit_transform() is applied only to training set
- **Transform validation/test**: validation and test sets are normalized using training statistics
- **Prevent data leakage**: Ensures the model doesn't see statistics from validation/test during training

### Reproducibility
- Fixed random seed ensures consistent results across runs
- Deterministic train/validation/test splits

## Customization

### Using Different Datasets

To use your own dataset:

```python
# Load your own CSV
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1).values
y = df["target"].values

# One-hot encode categorical columns
obj_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=obj_cols).astype(float)
```

### Adjusting Model Architecture for Different Input Sizes

To modify the MLP for different input dimensions:

```python
class CustomRegressionMLP(nn.Module):
    def __init__(self, input_size=10, hidden_sizes=[32, 16]):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for regression
        return x

# Usage with different dimensions
model = CustomRegressionMLP(input_size=20, hidden_sizes=[64, 32])
```

### Adjusting Hyperparameters

Modify `fenn.yaml`:

```yaml
train:
  seed: 42
  batch: 32        # Increase batch size
  epochs: 100      # Train longer
  lr: 5e-4         # Lower learning rate for fine-tuning
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
  val_loader=val_loader,  # validation occurs every epoch when provided
)
```

## Regression Metrics

The trainer automatically computes during validation:

- **MSE (Mean Squared Error)**: Average of squared differences between predictions and targets
- **MAE (Mean Absolute Error)**: Average of absolute differences between predictions and targets
- **R² Score**: Coefficient of determination (1.0 is perfect, 0.0 is baseline, negative is worse than baseline)

**Interpretation:**
- **R² = 1.0**: Perfect prediction
- **R² = 0.5**: Model explains 50% of variance in target
- **R² < 0.0**: Model performs worse than predicting mean value
- **MAE**: Average prediction error in original units
- **MSE**: Penalizes larger errors more than smaller ones

## Performance Monitoring

The trainer automatically logs during training:
- Training loss per epoch
- Validation MSE, MAE, R² score
- Early stopping events
- Best model checkpoint location

Example output:
```
Epoch 0. Mean Loss: 0.8234
Epoch 5. Validation Loss: 0.7125
Epoch 5. Validation MSE: 0.7125
Epoch 5. Validation MAE: 0.6234
Epoch 5. Validation R2: 0.8543
...
Early stopping triggered. No improvement for 5 epochs.
Best model checkpoint saved to ./checkpoints/checkpoint_best.pt
```

## Expected Results

On the Housing Price Prediction task:
- Training R²: ~0.85-0.92
- Validation R²: ~0.80-0.88
- Test R²: ~0.78-0.85

Results depend on:
- Random seed
- Train-validation-test split
- Hyperparameters
- Model architecture

## File Structure

```
mlp-regression/
├── main.py                       # Training script
├── modules/
│   ├── __init__.py              # Module imports
│   ├── model.py                 # RegressionMLP architecture
│   └── dataset.py               # RegressionDataset class
├── fenn.yaml                    # Configuration file
├── requirements.txt             # Python dependencies
└── mlp_regression.md           # This documentation
```

## Dependencies

Required packages:
- PyTorch
- scikit-learn (for preprocessing and metrics)
- numpy
- pandas
- kagglehub (for fetching datasets)
- FENN framework

Install with:
```bash
pip install torch scikit-learn numpy pandas kagglehub
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

## Comparing Regression with Classification

| Aspect | Classification | Regression |
|--------|----------------|-----------|
| Output | Discrete classes | Continuous values |
| Loss Function | CrossEntropyLoss / BCEWithLogitsLoss | MSELoss / L1Loss |
| Output Neurons | num_classes | 1 |
| Output Activation | Softmax / Sigmoid | None |
| Metrics | Accuracy, Precision, Recall, F1 | R², MSE, MAE |
| Prediction | Argmax / Threshold | Direct value |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch` size in fenn.yaml |
| Slow training | Enable GPU: set `device: cuda` |
| High test error | Try lower learning rate (5e-4) or more epochs |
| Overfitting | Add regularization or use smaller hidden layers |
| NaN loss | Check data normalization and learning rate |
| Poor predictions | Verify feature scaling is applied to test data |

## Extending to Larger Datasets

For production use with larger datasets:

1. **Avoid loading entire dataset into memory**:
```python
# Use parallel data loading
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4  # Parallel data loading
)
```

2. **Add data validation checks**:
```python
assert X_train.shape[1] == expected_features, "Feature mismatch"
assert not np.isnan(y_train).any(), "NaN values in targets"
```

3. **Monitor GPU memory**:
```bash
# Check GPU usage
nvidia-smi
```

## References

- [PyTorch Regression](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Kaggle Housing Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
- [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)
- [R-squared Score](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- FENN Framework Documentation
