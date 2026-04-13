# Vineet MLOps

A collection of machine learning experiments demonstrating various approaches to building, training, and deploying ML models using the Iris dataset.

## Project Overview

This project contains multiple experiments showcasing different ML workflows:
- Data exploration and visualization
- Model training and evaluation
- API deployment
- Experiment tracking with MLflow

---

## Experiments

### **Exp0: Logistic Regression with EDA (Jupyter Notebook)**

**File:** `logistic_regression_iris.ipynb`

**Purpose:** Exploratory Data Analysis and model training in an interactive notebook environment.

**Working:**
- Loads Iris dataset from seaborn
- Performs comprehensive EDA with visualizations (pairplots, histograms, correlation heatmap)
- Trains Logistic Regression model
- Evaluates performance with confusion matrix and classification report

**Key Components:**
- Data loading and inspection
- Statistical summary
- Pairplot visualization
- Distribution analysis
- Correlation analysis
- Model training and evaluation

---

### **Exp1: Logistic Regression (Python Script)**

**File:** `main.py`

**Purpose:** Python script version of Exp0 for non-interactive execution.

**Working:**
- Loads Iris dataset
- Performs basic EDA visualization
- Splits data (80/20 train-test)
- Trains Logistic Regression model
- Prints accuracy and classification report

**Key Components:**
- Dataset loading and preprocessing
- Train-test split
- Model training
- Performance metrics

---

### **Exp2: Basic Logistic Regression Training**

**Files:**
- `train.py` - Main training script
- `requirements.txt` - Project dependencies

**Purpose:** Simplified, production-ready script for training Logistic Regression.

**Working:**
- Loads and prepares Iris dataset
- Encodes target labels
- Splits data (80/20)
- Trains model with max_iter=200
- Outputs accuracy, classification report, and confusion matrix

**Key Components:**
- Minimal dependencies setup
- Clean model training pipeline
- Standardized evaluation metrics

---

### **Exp4: FastAPI Inference Server**

**Files:**
- `main.py` - FastAPI application
- `requirements.txt` - Dependencies (FastAPI, Pydantic, scikit-learn)

**Purpose:** REST API for real-time inference with the trained Logistic Regression model.

**Working:**
- Defines API input/output schemas using Pydantic
- Trains model on startup
- Provides `/health` endpoint for health checks
- Provides REST endpoints for predictions
- Returns predicted class and prediction probabilities

**Key Features:**
- Type-safe input validation
- Interactive API documentation (Swagger UI)
- Scalable inference server

---

### **Exp5: Dockerized FastAPI Inference Server**

**Files:**
- `main.py` - FastAPI application (same as Exp4)
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image configuration
- `docker-compose.yml` - Container orchestration
- `.dockerignore` - Exclude unnecessary files
- `README.md` - Detailed Docker documentation

**Purpose:** Containerized REST API for production deployment with Docker.

**Working:**
- Builds lightweight Docker image (Python 3.11 slim)
- Trains Logistic Regression model on container startup
- Serves predictions via FastAPI on port 8000
- Includes health checks and restart policies
- Supports easy scaling and deployment

**Key Features:**
- Docker containerization for consistency
- Docker Compose for easy orchestration
- Health checks for monitoring
- Production-ready configuration
- Multi-platform deployment support
- Automated restart on failure

**Quick Start:**
```bash
# Using Docker Compose
docker-compose up -d

# Or using Docker CLI
docker build -t iris-inference-api .
docker run -p 8000:8000 iris-inference-api
```

---

### **Exp6: MLflow Experiment Tracking**

**Files:**
├── Exp5/
│   ├── main.py                            # Dockerized FastAPI server
│   ├── requirements.txt                   # Python dependencies
│   ├── Dockerfile                         # Docker image config
│   ├── docker-compose.yml                 # Container orchestration
│   ├── .dockerignore                      # Exclude from build
│   └── README.md                          # Docker documentation
- `main.py` - Main training script with MLflow integration
- `mlruns/` - MLflow runs storage (local file system)

**Purpose:** Production-grade experiment tracking and model logging using MLflow.

**Working:**
- Sets up MLflow tracking URI and experiment
- Enables autolog for scikit-learn models
- Trains Logistic Regression with feature scaling
- Automatically logs:
  - Model parameters
  - Training metrics (accuracy, F1, precision, recall, ROC-AUC, etc.)
  - Input datasets and artifacts
  - Model artifacts and estimator details

**MLflow Artifacts Tracked:**
- Model serialization
- Parameter configurations
- Evaluation metrics
- Input/output datasets
- Conda environment and python requirements

---

## File Structure

```
Vineet Mlops/
├── README.md                              # Project documentation
├── Exp0/
│   └── logistic_regression_iris.ipynb     # Jupyter notebook (EDA + training)
├── Exp1/
│   └── main.py                            # Python script version
├── Exp2/
│   ├── train.py                           # Simplified training script
│   └── requirements.txt                   # Dependencies
├── Exp4/
│   ├── main.py                            # FastAPI inference server
│   └── requirements.txt                   # Dependencies
└── Exp6/
    ├── main.py                            # MLflow integrated training
    ├── requirements.txt                   # Dependencies
    └── mlruns/                            # MLflow runs & artifacts storage
        ├── 0/                             # Metadata
        ├── 912011795011189900/            # Experiment runs
        │   ├── meta.yaml
        │   ├── 584134e7c4a44df783ff91a2c6684167/  # Run artifacts
        │   │   ├── artifacts/
        │   │   ├── inputs/
        │   │   ├── metrics/
        │   │   ├── outputs/
        │   │   ├── params/
        │   │   └── tags/
        │   ├── datasets/
        │   ├── models/
        └── models/                        # Logged models
└── Exp7/
    ├── prepare_data.py                    # Data preparation stage
    ├── train_model.py                     # Model training stage
    ├── evaluate_model.py                  # Model evaluation stage
    ├── dvc.yaml                           # DVC pipeline definition
    ├── params.yaml                        # Hyperparameters
    ├── requirements.txt                   # Python dependencies
    ├── .gitignore                         # Git ignore rules
    ├── README.md                          # DVC documentation
    ├── dvc.lock                           # DVC lock file (auto-generated)
    ├── data/                              # Datasets (auto-generated)
    ├── models/                            # Trained models (auto-generated)
    └── metrics/                           # Evaluation metrics (auto-generated)
```

---

### **Exp7: DVC ML Pipeline**

**Files:**
- `prepare_data.py` - Data preparation stage
- `train_model.py` - Model training stage
- `evaluate_model.py` - Model evaluation stage
- `dvc.yaml` - Pipeline definition
- `params.yaml` - Hyperparameters
- `requirements.txt` - Dependencies
- `README.md` - Comprehensive documentation

**Purpose:** Reproducible ML pipeline using DVC (Data Version Control) for version tracking and workflow automation.

**Working:**
- Defines multi-stage data → train → evaluate pipeline
- Tracks all data, models, and metrics with DVC
- Parameterizes model hyperparameters in `params.yaml`
- Automatically handles dependencies and re-runs
- Ensures reproducibility across environments

**Key Benefits:**
- **Reproducibility**: Same params + data = same results
- **Version Control**: Track entire ML workflow
- **Automation**: Automatic stage dependencies
- **Collaboration**: Clean git history with DVC tracking
- **Scalability**: Easy to scale with remote storage

**Quick Start:**
```bash
cd Exp7
pip install -r requirements.txt
dvc repro           # Run full pipeline
dvc metrics show    # View results
```

---

## Getting Started

1. **Exp0:** Open `Exp0/logistic_regression_iris.ipynb` in Jupyter
2. **Exp1/Exp2:** Run `python main.py` or `python train.py`
3. **Exp4:** Run FastAPI server:
   ```bash
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
4. **Exp6:** Run MLflow tracking:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
   Then view results: `mlflow ui --backend-store-uri ./mlruns`

---

## Dataset

All experiments use the **Iris dataset** from seaborn:
- **Features:** Sepal length, sepal width, petal length, petal width
- **Target:** Species (setosa, versicolor, virginica)
- **Samples:** 150 (150 after removing duplicates in Exp6)

