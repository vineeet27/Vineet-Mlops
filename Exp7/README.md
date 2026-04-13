# Exp7: DVC ML Pipeline for Iris Classification

A production-grade Machine Learning pipeline using **DVC (Data Version Control)** for reproducible ML workflows.

## Overview

This experiment demonstrates:
- **Reproducible ML Pipeline** - DVC tracks data, models, and metrics
- **Version Control** - Track all ML artifacts and dependencies
- **Parameterization** - Manage hyperparameters via `params.yaml`
- **Multi-stage Pipeline** - Data → Training → Evaluation
- **Metrics Tracking** - Automatic logging of model performance

## Pipeline Architecture

```
prepare_data.py → dvc.yaml → train_model.py → evaluate_model.py
     ↓               ↓             ↓                ↓
  data/          params.yaml    models/          metrics/
```

### Pipeline Stages

#### 1. **prepare** - Data Preparation
- Loads Iris dataset from seaborn
- Removes duplicates
- Encodes target labels
- Splits into train (80%) and test (20%) sets
- Saves all preprocessed data to `data/` directory

**Output:**
- `data/iris_full.csv` - Full dataset
- `data/X_train.csv` - Training features
- `data/X_test.csv` - Test features
- `data/y_train.csv` - Training labels
- `data/y_test.csv` - Test labels
- `data/label_encoding.json` - Label mapping

#### 2. **train** - Model Training
- Loads training data
- Creates preprocessing pipeline (StandardScaler)
- Trains Logistic Regression model
- Saves trained model to `models/model.pkl`

**Parameters:**
- `max_iter`: 200
- `solver`: 'lbfgs'
- `multi_class`: 'multinomial'
- `random_state`: 42

#### 3. **evaluate** - Model Evaluation
- Loads trained model and test data
- Generates predictions
- Calculates metrics (accuracy, precision, recall, F1, ROC-AUC)
- Saves metrics and confusion matrix

**Output:**
- `metrics/metrics.json` - Performance metrics
- `metrics/confusion_matrix.json` - Confusion matrix
- `metrics/classification_report.txt` - Detailed classification report

## File Structure

```
Exp7/
├── prepare_data.py          # Data preparation stage
├── train_model.py           # Model training stage
├── evaluate_model.py        # Model evaluation stage
├── dvc.yaml                 # Pipeline definition
├── params.yaml              # Hyperparameters
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── dvc.lock                # Lock file (auto-generated)
├── data/                   # Data directory (auto-generated)
│   ├── iris_full.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   └── label_encoding.json
├── models/                 # Models directory (auto-generated)
│   └── model.pkl
└── metrics/                # Metrics directory (auto-generated)
    ├── metrics.json
    ├── confusion_matrix.json
    └── classification_report.txt
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize DVC (One-time Setup)

```bash
dvc init
git add .dvc .gitignore dvc.yaml params.yaml
git commit -m "Initialize DVC pipeline"
```

### 3. Run the Pipeline

**Run all stages:**
```bash
dvc repro
```

**Run specific stage:**
```bash
dvc repro prepare
dvc repro train
dvc repro evaluate
```

### 4. View Pipeline Status

```bash
dvc dag
```

Example output:
```
    prepare
      ↓
    train
      ↓
  evaluate
```

### 5. View Metrics

```bash
dvc metrics show
```

### 6. Modify and Re-run

Edit `params.yaml` to change hyperparameters:
```yaml
train:
  model_params:
    max_iter: 300  # Changed from 200
    random_state: 42
    solver: 'lbfgs'
    multi_class: 'multinomial'
```

Then re-run:
```bash
dvc repro
```

## DVC Concepts

### **dvc.yaml**
Defines the pipeline stages with dependencies and outputs.

### **params.yaml**
Centralized hyperparameter management. Changes trigger automatic re-training.

### **dvc.lock**
Auto-generated lock file that tracks exact versions of all artifacts.

### **Reproducibility**
- Same params + same data = same model
- All stages are automatically tracked
- Easy to revert to previous versions

## Advantages Over Manual Workflows

| Manual | DVC |
|--------|-----|
| Run scripts manually | Automatic dependency tracking |
| Hard to track versions | Every stage is versioned |
| Easy to skip steps | Ensures reproducibility |
| Parameter changes hidden | Centralized params.yaml |
| Difficult to collaborate | Clean git history |

## Advanced Usage

### View Pipeline DAG (Dependency Graph)

```bash
dvc dag
```

### Compare Metrics Across Runs

```bash
dvc metrics diff
```

### Push Data to Remote Storage

```bash
dvc remote add -d myremote /path/to/remote
dvc push
```

### Visualize Metrics

```bash
dvc plots show
```

## Integration with Git

The pipeline is fully git-integrated:

```bash
# Commit pipeline definition
git add .
git commit -m "Add DVC ML pipeline"

# DVC tracks data and models separately
# Only dvc.yaml, params.yaml, dvc.lock are in git
```

### .gitignore Rules
- `/data` - Large datasets not in git
- `/models` - Model files tracked by DVC
- `/metrics` - Generated metrics
- `dvc.lock` - Auto-generated dependency lock

## Troubleshooting

### "DVC not found" error
```bash
pip install dvc
```

### Force re-run all stages
```bash
dvc repro --force
```

### Clear all outputs and re-run
```bash
dvc repro --force --rerun
```

### View pipeline structure
```bash
dvc dag --ascii
```

## Production Deployment

For production pipelines:

1. **Use DVC Pipelines** with scheduled runs
2. **CI/CD Integration** - Trigger with git push
3. **Data Versioning** - Track data changes
4. **Model Registry** - Manage model versions
5. **Remote Storage** - S3, Azure Blob, GCS

## Next Steps

- Add input validation
- Implement hyperparameter tuning
- Add data quality checks
- Integrate with DVC Studio for visualization
- Deploy model using Docker (Exp5 style)

## References

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorials](https://dvc.org/learn)
- [ML Pipeline Best Practices](https://dvc.org/blog)
