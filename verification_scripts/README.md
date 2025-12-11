# Verification Scripts for Team 18 Final Report

This folder contains scripts that verify all quantitative claims made in the final report. Each script can be run independently to reproduce the reported results.

## Purpose

These verification scripts address academic integrity by providing reproducible evidence for:
- Training times (SVM: 1.12s (~1s), RF: 1.56s (~2s), LR: 1.53s (~2s), LSTM: ~14.82s (~15s))
- Hyperparameter tuning results (C=1.0 for SVM, n_estimators=100 for RF, max_iter=1,000, units=16)
- XAI consensus rates (78.15% agreement between LIME and SHAP)

## Setup

### Prerequisites

We already included all dependencies in our requirements.txt file. The scripts use some from that list, please install those if not already done.


### Activate Environment

```bash
cd "/verification_scripts"
source ../ML-Spam-Email-Detection/venv/bin/activate  # Linux/Mac
# OR
../ML-Spam-Email-Detection/venv/Scripts/activate  # Windows
```

### Required Files

The scripts expect the following files to exist:
- `../ML-Spam-Email-Detection/Data/spam_ham_dataset.csv` (dataset)
- `../ML-Spam-Email-Detection/models/svm_model.pkl` (for XAI consensus script)
- `../ML-Spam-Email-Detection/models/svm_vectorizer.pkl` (for XAI consensus script)

If the model files don't exist, run the SVM training script first:
```bash
cd "../ML-Spam-Email-Detection/Train Scripts"
python SVM.py
```

## Running Verification Scripts

### 1. Training Time Verification

**Purpose:** Proves training times match report claims.

**Command:**
```bash
python timing_verification.py
```

**Expected output:**
```
For each of the models (SVM, Random Forest, Logistic Regression, TensorFloe LSTM), if our report claims match the expected value range, we get "✓ VERIFIED" otherwise, we're notified as well through "✗ DEVIATION".
```

**Output file:** `timing_results.txt`

**What it does:**
- Loads and preprocesses the dataset
- Trains each model 5 times independently
- Measures elapsed time for each run
- Reports mean ± standard deviation
- Verifies results match report claims (with some margin of increasing or decreasing tolerance)

---

### 2. Hyperparameter Tuning Verification

**Purpose:** Proves hyperparameters were tested and optimal values were selected.

**Command:**
```bash
python hyperparameter_tuning_verification.py
```

**Expected output:**
```
The values would be:
C=1.0,  n_est=50, n_est=100, iter=1000, units=16
```

**Output file:** `hyperparameter_results.txt`

**What it does:**
- Tests multiple hyperparameter values for each model
- Uses 5-fold cross-validation for robust evaluation
- Evaluates on held-out test set
- Identifies optimal values based on test accuracy
- Verifies selected values match report claims

---

### 3. XAI Consensus Verification

**Purpose:** Proves 78.15% consensus between LIME and SHAP explanations.

**Command:**
```bash
python xai_consensus_verification.py
```

**Expected output:**
```
=== Overall Results ===
Mean consensus: 78.15%
Status: ✓ VERIFIED
```

**Output file:** `consensus_results.txt`

**What it does:**
- Loads the trained SVM model and vectorizer
- Selects 20 representative emails (10 spam, 10 ham)
- Generates LIME explanations for each email
- Calculates SHAP-aligned feature importance
- Measures consensus rate (|LIME ∩ SHAP| / |LIME ∪ SHAP|)
- Reports mean ± standard deviation
- Verifies results match report claims (70-86% range)

---

## Running All Scripts

To run all verification scripts sequentially:

```bash
# Timing verification
echo "Running timing verification..."
python timing_verification.py

# Hyperparameter tuning verification
echo "Running hyperparameter tuning verification..."
python hyperparameter_tuning_verification.py

# XAI consensus verification
echo "Running XAI consensus verification..."
python xai_consensus_verification.py

echo "All verifications complete!"
```

For questions or issues with verification scripts, please refer to our main project repository:
https://github.com/andrewEdson/ML-Spam-Email-Detection
