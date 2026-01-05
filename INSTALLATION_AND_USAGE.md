# Installation & Usage Guide: Pre-Crash Intelligence System

Complete step-by-step instructions to set up and run the Pre-Crash Intelligence System for Two-Wheelers.

---

## Prerequisites

- **Python 3.8+** installed on your system
- **macOS/Linux/Windows** with terminal/command prompt access
- **~5 GB disk space** for datasets and models
- **Virtual environment** (recommended)

---

## Step 1: Environment Setup

### 1.1 Navigate to Project Directory

```bash
cd "/Users/kartikeyamishra/Downloads/kartikeya ev"
```

### 1.2 Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows (Command Prompt):
.venv\Scripts\activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 1.3 Upgrade pip

```bash
pip install --upgrade pip
```

### 1.4 Install Required Packages

```bash
pip install -r requirements.txt
```

**Packages installed:**
- `numpy` — numerical computing
- `pandas` — data manipulation
- `scikit-learn` — machine learning algorithms
- `tensorflow` — deep learning (for neural network model)
- `matplotlib` & `seaborn` — visualization
- `scipy` — scientific computing
- `imbalanced-learn` — SMOTE for class balancing
- `joblib` — model serialization

**Installation time:** ~3-5 minutes

---

## Step 2: Project Structure Verification

Verify that all required directories and files are present:

```bash
# Check main files
ls -la *.py  # Should show: main.py, compare_models.py, demo_predictions.py, early_warning_analysis.py

# Check datasets
ls -la dataset/
ls -la dataset2/
ls -la dataset3/

# Check source code
ls -la src/
# Should contain: data_loader.py, feature_extractor.py, models.py, predictor.py, eda_analysis.py

# Check output directory exists
ls -la output/
```

**Expected structure:**
```
kartikeya ev/
├── src/                           # Source code
│   ├── data_loader.py
│   ├── feature_extractor.py
│   ├── models.py
│   ├── predictor.py
│   └── eda_analysis.py
├── dataset/                       # Dataset 1: Corrected data
├── dataset2/                      # Dataset 2: Falls scenarios
├── dataset3/                      # Dataset 3: Extreme maneuvers
├── models/                        # Saved models (created during training)
├── output/                        # Results and visualizations
├── main.py                        # Main pipeline
├── compare_models.py              # Model comparison
├── demo_predictions.py            # Demo script
├── early_warning_analysis.py      # Early warning analysis
└── requirements.txt               # Dependencies
```

---

## Step 3: Run the Complete Pipeline

### 3.1 Train All Models (First Time)

This is the main entry point. It will:
- Load data from 3 datasets
- Perform exploratory data analysis (EDA)
- Extract 111 features
- Train 7 machine learning models
- Generate visualizations and reports

```bash
python main.py --mode train --skip-eda --base-path "/Users/kartikeyamishra/Downloads/kartikeya ev"
```

**Options:**
- `--mode train` — run full training pipeline
- `--skip-eda` — skip time-consuming EDA plots (remove if you want full analysis)
- `--base-path` — path to project directory
- `--test-size 0.2` — test split ratio (default: 0.2 = 20%)
- `--window-size 1.0` — feature window size in seconds (default: 1.0)
- `--step-size 0.5` — sliding step in seconds (default: 0.5)

**Expected runtime:** 3-8 minutes (depending on system)

**Output:** 
```
Models saved to: models/
  ✓ random_forest_model.pkl
  ✓ gradient_boosting_model.pkl
  
Reports saved to: output/
  ✓ performance_report.json
  ✓ feature_importance.csv
  ✓ feature_info.json
  
Visualizations saved to: output/visualizations/
  ✓ model_comparison.png
  ✓ confusion_matrix.png
  ✓ roc_curve.png
  ✓ feature_importance_top15.png
  ... (and more)
```

### 3.2 Run Demo with Trained Model

After training, run live predictions on sample crash data:

```bash
python main.py --mode demo --base-path "/Users/kartikeyamishra/Downloads/kartikeya ev"
```

**Or use dedicated demo script:**

```bash
python demo_predictions.py
```

**Output:**
- Real-time crash probability estimates
- Prediction confidence scores
- Sample predictions on crash and safe events

---

## Step 4: Compare All Seven Models

Systematically benchmark and compare all 7 machine learning algorithms:

```bash
python compare_models.py
```

**This runs:**
1. Random Forest (91.77% accuracy)
2. Gradient Boosting (89.24% accuracy)
3. Neural Network (89.81% accuracy)
4. Decision Tree (85.96% accuracy)
5. Support Vector Machine (76.13% accuracy)
6. k-Nearest Neighbors (78.16% accuracy)
7. Logistic Regression (68.25% accuracy)

**Expected runtime:** 2-5 minutes

**Output:**
- Comparison table with all metrics
- Confusion matrices for each model
- Visualizations comparing performance
- CSV file with results: `output/model_comparison.csv`

---

## Step 5: Analyze Early Warning Capability

Validate that the system can predict crashes 1-3 seconds in advance:

```bash
python early_warning_analysis.py
```

**This analyzes:**
- Temporal lead time for each crash detection
- Distribution of warning times
- Performance across crash types
- Cross-dataset validation

**Expected runtime:** 1-2 minutes

**Output:**
```
Early Warning Analysis Results:
  Median warning time: 2.1 seconds
  5th percentile: 1.5 seconds
  95th percentile: 3.0 seconds
  Detection rate: 93.51%
  
Visualizations:
  ✓ early_warning_capability.png
  ✓ scenario_timelines.png
```

---

## Step 6: View Results

### 6.1 Check Performance Metrics

```bash
# View detailed performance report
cat output/performance_report.json

# View feature importance
head -20 output/feature_importance.csv
```

### 6.2 View Visualizations

All visualizations are saved in `output/visualizations/`:

```bash
# List all generated plots
ls -la output/visualizations/

# Open plots (macOS)
open output/visualizations/model_comparison.png
open output/visualizations/confusion_matrix.png
open output/visualizations/roc_curve.png
open output/visualizations/feature_importance_top15.png
open output/visualizations/early_warning_capability.png
```

**Key visualizations:**
1. `model_comparison.png` — Accuracy/recall across 7 models
2. `confusion_matrix.png` — Crash detection vs false positives
3. `roc_curve.png` — ROC curves for all models
4. `feature_importance_top15.png` — Most predictive features
5. `early_warning_capability.png` — Warning time distribution
6. `dataset_summary.png` — Dataset composition
7. `class_distribution.png` — Crash vs safe event balance
8. `sensor_correlation_matrix.png` — IMU channel relationships

---

## Quick Start (Single Command)

If you want to run everything with one command:

```bash
# 1. Setup environment
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. Run full pipeline
python main.py --mode train --skip-eda --base-path "/Users/kartikeyamishra/Downloads/kartikeya ev"

# 3. Run analysis
python early_warning_analysis.py && python compare_models.py

# 4. View results
ls -la output/visualizations/
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Ensure you're in the correct directory
cd "/Users/kartikeyamishra/Downloads/kartikeya ev"

# Verify src/ exists
ls src/

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: "No such file or directory: dataset/"

**Solution:**
```bash
# Verify datasets are present
ls dataset/
ls dataset2/
ls dataset3/

# If missing, extract from zip files (if present)
unzip dataset.zip
unzip dataset2.zip
unzip dataset3.zip
```

### Issue: "TensorFlow not found" or GPU errors

**Solution:**
```bash
# Reinstall TensorFlow for CPU
pip install --force-reinstall tensorflow-cpu

# Or use the GPU version (requires CUDA)
pip install tensorflow
```

### Issue: Out of memory errors

**Solution:**
```bash
# Reduce batch size in feature extraction
# Edit main.py and modify the feature extraction parameters
# Or run on a machine with more RAM (16GB+ recommended)
```

### Issue: Model training takes too long

**Solution:**
```bash
# Skip EDA (as already done)
python main.py --mode train --skip-eda --base-path "/path/to/project"

# Reduce test size for faster iteration
python main.py --mode train --skip-eda --test-size 0.1 --base-path "/path/to/project"
```

---

## Expected Performance Metrics

After running the pipeline, you should see:

```
MODEL COMPARISON RESULTS:
┌─────────────────────┬──────────┬───────────┬────────┬────────┬─────────┐
│ Model               │ Accuracy │ Precision │ Recall │ F1     │ AUC     │
├─────────────────────┼──────────┼───────────┼────────┼────────┼─────────┤
│ Random Forest       │ 91.77%   │ 96.43%    │ 93.51% │ 94.94% │ 96.41%  │
│ Gradient Boosting   │ 89.24%   │ 93.53%    │ 93.45% │ 93.49% │ 93.70%  │
│ Neural Network      │ 89.81%   │ 95.42%    │ 92.10% │ 93.73% │ 94.64%  │
│ Decision Tree       │ 85.96%   │ 95.07%    │ 87.55% │ 91.15% │ 86.80%  │
│ K-Nearest Neighbors │ 78.16%   │ 97.25%    │ 75.73% │ 85.15% │ 89.06%  │
│ SVM (RBF)           │ 76.13%   │ 96.76%    │ 73.58% │ 83.59% │ 89.22%  │
│ Logistic Regression │ 68.25%   │ 95.54%    │ 64.61% │ 77.09% │ 83.03%  │
└─────────────────────┴──────────┴───────────┴────────┴────────┴─────────┘

BEST MODEL: Random Forest
  Crash Detection Rate: 93.51% (562/601 crashes detected)
  Early Warning Time: 1.5–3.0 seconds (Median: 2.1 sec)
  Inference Latency: <5 ms per prediction
```

---

## Next Steps

After successful model training and validation:

1. **Deploy to Mobile**: Use TensorFlow Lite to convert models for Android/iOS
2. **Real-time Integration**: Integrate with smartphone IMU sensors
3. **Edge Deployment**: Deploy on Raspberry Pi or ESP32
4. **Field Testing**: Validate on real motorcycles in controlled environments
5. **Model Refinement**: Retrain with additional data for improved accuracy

---

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Main pipeline orchestrator |
| `compare_models.py` | Benchmark 7 ML algorithms |
| `demo_predictions.py` | Demo predictions on sample data |
| `early_warning_analysis.py` | Analyze 1-3 second early warning capability |
| `src/data_loader.py` | Load and preprocess multi-dataset IMU data |
| `src/feature_extractor.py` | Extract 111 engineered features |
| `src/models.py` | Train and evaluate ML models |
| `src/predictor.py` | Real-time prediction interface |
| `src/eda_analysis.py` | Generate exploratory visualizations |

---

## Contact & Support

For issues or questions:
1. Check the README.md for project overview
2. Review error messages in console output
3. Verify all dataset files are present and accessible
4. Ensure Python 3.8+ and all dependencies installed correctly

---

**Last Updated:** January 5, 2026  
**System:** Pre-Crash Intelligence System for Two-Wheelers  
**Version:** 1.0 Production