# ML Spam Email Detection

A comprehensive machine learning project comparing different algorithms for spam email classification. This project implements and evaluates multiple ML approaches to determine the most effective method for detecting spam emails.

## Project Overview

This project tests and compares four different machine learning algorithms for spam email detection. Through rigorous evaluation, we identify the best-performing model and provide explainable AI insights using LIME and advanced feature importance analysis.

## Project Structure

```
ML-Spam-Email-Detection/
├── Data/
│   └── spam_ham_dataset.csv
├── Train Scripts/
│   ├── Tensorflow.py          # LSTM neural network
│   ├── SVM.py                  # Support Vector Machine
│   ├── RandomForest.py         # Random Forest classifier
│   └── LogisticRegression.py   # Logistic Regression
├── Evaluation/
│   ├── ModelComparison.ipynb   # Comprehensive model comparison
│   └── LLM_Explanation.py      # AI-powered explanations with LIME
├── models/                      # Trained models and vectorizers
└── requirements.txt
```

## Implemented Models

### 1. Support Vector Machine (SVM) - **Best Model (99% Accuracy)**

- **File**: `Train Scripts/SVM.py`
- **Features**: Linear kernel with TF-IDF vectorization
- **Performance**: 99% accuracy with excellent precision and recall
- **Best for**: Production deployment due to speed and accuracy

### 2. Random Forest

- **File**: `Train Scripts/RandomForest.py`
- **Features**: Ensemble of 100 decision trees
- **Performance**: 97% accuracy
- **Best for**: Feature importance analysis

### 3. Logistic Regression

- **File**: `Train Scripts/LogisticRegression.py`
- **Features**: Simple linear classifier
- **Performance**: 96% accuracy
- **Best for**: Fast training and interpretability

### 4. TensorFlow LSTM

- **File**: `Train Scripts/Tensorflow.py`
- **Architecture**: Sequential model with Embedding, LSTM, and Dense layers
- **Performance**: 96% accuracy
- **Features**: Word cloud visualization, early stopping, learning rate reduction
- **Best for**: Sequential pattern recognition

## Dataset

- **Source**: `Data/spam_ham_dataset.csv`
- **Content**: Email text data with spam/ham labels
- **Preprocessing**:
  - Balanced dataset (equal spam and ham samples)
  - Punctuation removal
  - Stopword filtering
  - TF-IDF vectorization (sklearn models)
  - Tokenization and padding (TensorFlow model)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/andrewEdson/ML-Spam-Email-Detection.git
   cd ML-Spam-Email-Detection
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Set up OpenAI API for LLM explanations:
   - Create a `.env` file in the root directory
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

## Usage

### Training Models

Train individual models:

```bash
python "Train Scripts/SVM.py"
python "Train Scripts/RandomForest.py"
python "Train Scripts/LogisticRegression.py"
python "Train Scripts/Tensorflow.py"
```

All trained models are automatically saved to the `models/` directory.

### Model Evaluation

Compare all models in the Jupyter notebook:

```bash
jupyter notebook Evaluation/ModelComparison.ipynb
```

The notebook provides:

- Side-by-side performance comparison
- Classification reports for each model
- Comprehensive analysis and recommendation

### Explainable AI

Get AI-powered explanations for predictions:

```bash
python Evaluation/LLM_Explanation.py
```

This script:

- Uses LIME for local interpretability
- Analyzes feature importance
- Generates natural language explanations via ChatGPT
- Identifies features both methods agree on

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- nltk
- wordcloud
- tensorflow
- scikit-learn
- joblib
- lime
- python-dotenv
- openai

## Results Summary

| Model               | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score (Avg) |
| ------------------- | -------- | --------------- | ------------ | -------------- |
| **SVM**             | **99%**  | **0.99**        | **0.99**     | **0.99**       |
| Random Forest       | 97%      | 0.97            | 0.97         | 0.97           |
| Logistic Regression | 96%      | 0.97            | 0.96         | 0.96           |
| TensorFlow LSTM     | 96%      | 0.96            | 0.97         | 0.96           |

## Key Features

✅ Multiple ML algorithms implemented and compared  
✅ Automated model training and saving  
✅ Comprehensive evaluation metrics  
✅ Explainable AI with LIME  
✅ LLM-powered natural language explanations  
✅ Production-ready model recommendations  
✅ Jupyter notebook for interactive analysis
