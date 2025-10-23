# ML Spam Email Detection

A machine learning project comparing different algorithms for spam email classification. This project evaluates various ML approaches to determine the most effective method for detecting spam emails.

## Project Overview

This project focuses on testing and comparing different machine learning algorithms for spam email detection, starting with TensorFlow/Keras neural networks. The goal is to identify which algorithm performs best for classifying emails as spam or ham (legitimate emails).

## Current Implementation

### TensorFlow LSTM Model

- **File**: `Tensorflow.py`
- **Architecture**: Sequential model with Embedding layer, LSTM layer, and Dense layers
- **Features**:
  - Text preprocessing (punctuation removal, stopword filtering)
  - Data balancing to handle class imbalance
  - Word cloud visualization for spam vs ham analysis
  - LSTM neural network for sequence classification
  - Early stopping and learning rate reduction callbacks

## Dataset

- **Source**: `Data/spam_ham_dataset.csv`
- **Content**: Email text data with spam/ham labels
- **Preprocessing**: Balanced dataset to ensure equal representation of spam and ham emails

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the TensorFlow model:
   ```bash
   python Tensorflow.py
   ```

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- nltk
- wordcloud
- tensorflow
- scikit-learn

## Future Work

This project will expand to include additional algorithms such as:

- Support Vector Machines (SVM)
- Random Forest
- Logistic Regression

## Results

The TensorFlow LSTM model provides baseline performance metrics for comparison with future algorithm implementations.
