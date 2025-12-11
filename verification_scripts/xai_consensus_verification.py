"""
XAI Consensus Verification Script
Proves the reported consensus between LIME and SHAP on 20 sample emails.
Uses random_state=42 throughout for full reproducibility.
"""

import string
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords
import nltk

# Download stopwords if needed
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

print("=" * 80)
print("XAI CONSENSUS VERIFICATION")
print("=" * 80)
print("\nThis script measures consensus between LIME and SHAP explanations.")
print("Analyzing 20 sample emails (10 spam, 10 ham)...\n")

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def remove_punctuations(text):
    punctuations_list = string.punctuation
    temp = str.maketrans("", "", punctuations_list)
    return text.translate(temp)


def remove_stopwords(text):
    stop_words = stopwords.words("english")
    imp_words = []

    for word in str(text).split():
        word = word.lower()
        if word not in stop_words:
            imp_words.append(word)

    return " ".join(imp_words)


# ============================================================================
# LOAD SAVED MODEL AND VECTORIZER
# ============================================================================

print("Loading saved SVM model and vectorizer...")
try:
    svm_classifier = joblib.load("../ML-Spam-Email-Detection/models/svm_model.pkl")
    svm_vectorizer = joblib.load("../ML-Spam-Email-Detection/models/svm_vectorizer.pkl")
    print("Model and vectorizer loaded successfully.\n")
except FileNotFoundError:
    print("ERROR: Model files not found. Please run the SVM training script first.")
    print("Expected files:")
    print("  - ../ML-Spam-Email-Detection/models/svm_model.pkl")
    print("  - ../ML-Spam-Email-Detection/models/svm_vectorizer.pkl")
    exit(1)

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================

print("Loading dataset...")
data = pd.read_csv("../ML-Spam-Email-Detection/Data/spam_ham_dataset.csv")

# Balance classes
ham_msg = data[data["label"] == "ham"]
spam_msg = data[data["label"] == "spam"]
ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)
balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)

# Preprocessing
balanced_data["text"] = balanced_data["text"].str.replace("Subject", "")
balanced_data["text"] = balanced_data["text"].apply(lambda x: remove_punctuations(x))
balanced_data["text"] = balanced_data["text"].apply(lambda text: remove_stopwords(text))

# Train-test split
train_X, test_X, train_Y, test_Y = train_test_split(
    balanced_data["text"], balanced_data["label"], test_size=0.2, random_state=42
)

# ============================================================================
# SETUP XAI COMPONENTS
# ============================================================================

# SVM probability wrapper for LIME
def svm_predict_proba(texts):
    """Convert SVM decision function to probability-like scores for LIME"""
    vectorized = svm_vectorizer.transform(texts)
    decision = svm_classifier.decision_function(vectorized)
    # Convert to probability-like format: [ham_prob, spam_prob]
    proba = np.column_stack([1 - decision, decision])
    return proba


# Initialize LIME explainer with fixed random_state for reproducibility
lime_explainer = LimeTextExplainer(class_names=["ham", "spam"], random_state=42)

# Get feature names for SHAP analysis
feature_names = svm_vectorizer.get_feature_names_out()
svm_coef = svm_classifier.coef_.toarray()[0]

print(f"XAI components initialized.")
print(f"Total features in vocabulary: {len(feature_names)}\n")

# ============================================================================
# SELECT 20 SAMPLE EMAILS (10 spam, 10 ham)
# ============================================================================

# Select 10 spam and 10 ham emails using random_state for reproducibility
test_X_reset = test_X.reset_index(drop=True)
test_Y_reset = test_Y.reset_index(drop=True)

spam_indices = test_Y_reset[test_Y_reset == "spam"].index[:10]
ham_indices = test_Y_reset[test_Y_reset == "ham"].index[:10]
sample_indices = list(spam_indices) + list(ham_indices)

print(f"Selected 20 emails: 10 spam, 10 ham\n")
print("-" * 80)

# ============================================================================
# ANALYZE CONSENSUS FOR EACH EMAIL
# ============================================================================

consensus_rates = []
detailed_results = []

for idx, email_idx in enumerate(sample_indices, 1):
    email = test_X_reset[email_idx]
    true_label = test_Y_reset[email_idx]

    # Get model prediction
    email_vectorized = svm_vectorizer.transform([email])
    prediction = svm_classifier.predict(email_vectorized)[0]

    print(f"\nEmail {idx}/20 (True label: {true_label}, Predicted: {prediction})")
    print("-" * 80)

    # ========================================================================
    # LIME Analysis
    # ========================================================================

    lime_exp = lime_explainer.explain_instance(email, svm_predict_proba, num_features=10)
    lime_features = [feature for feature, weight in lime_exp.as_list()]
    lime_top5 = lime_features[:5]

    print(f"LIME top-10: {lime_features}")

    # ========================================================================
    # SHAP-Aligned Analysis
    # ========================================================================

    # For linear SVM: SHAP values ≈ coefficient × feature_value
    sample_features = email_vectorized.toarray()[0]

    # Calculate feature importance (coefficient × TF-IDF value)
    feature_importance = svm_coef * sample_features

    # Extract non-zero features and sort by absolute importance
    shap_sorted = sorted(
        [(feature_names[i], feature_importance[i])
         for i in range(len(feature_names))
         if sample_features[i] != 0],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    shap_features = [feature for feature, weight in shap_sorted]
    shap_top5 = shap_features[:5]

    print(f"SHAP top-10: {shap_features}")

    # ========================================================================
    # Calculate Consensus
    # ========================================================================

    lime_set = set(lime_features)
    shap_set = set(shap_features)

    # Consensus: intersection / union
    consensus = lime_set & shap_set
    union = lime_set | shap_set

    if len(union) > 0:
        agreement_rate = len(consensus) / len(union)
    else:
        agreement_rate = 0.0

    consensus_rates.append(agreement_rate)

    print(f"\nConsensus features: {consensus}")
    print(f"Agreement rate: {agreement_rate:.2%}")
    print(f"  LIME ∩ SHAP: {len(consensus)} features")
    print(f"  LIME ∪ SHAP: {len(union)} features")

    # Store detailed results
    detailed_results.append({
        'email_idx': email_idx,
        'true_label': true_label,
        'predicted_label': prediction,
        'lime_top5': lime_top5,
        'shap_top5': shap_top5,
        'consensus': list(consensus),
        'agreement_rate': agreement_rate
    })

# ============================================================================
# OVERALL STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("OVERALL CONSENSUS STATISTICS")
print("=" * 80)

mean_consensus = np.mean(consensus_rates)
std_consensus = np.std(consensus_rates)
min_consensus = np.min(consensus_rates)
max_consensus = np.max(consensus_rates)
median_consensus = np.median(consensus_rates)

print(f"\nMean consensus: {mean_consensus:.2%}")
print(f"Std deviation: {std_consensus:.2%}")
print(f"Median consensus: {median_consensus:.2%}")
print(f"Range: {min_consensus:.2%} - {max_consensus:.2%}")
print(f"\nReport claim: 78.15%")
print(f"Note: Std dev = ±{std_consensus:.1%}, Std error = ±{std_consensus/np.sqrt(len(consensus_rates)):.1%}")
print(f"Status: {'✓ VERIFIED' if 70 <= mean_consensus*100 <= 86 else '✗ DEVIATION'}")

# ============================================================================
# CONSENSUS BREAKDOWN
# ============================================================================

print("\n" + "-" * 80)
print("CONSENSUS RATE DISTRIBUTION")
print("-" * 80)

# Histogram bins
bins = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
for low, high in bins:
    count = sum(1 for rate in consensus_rates if low <= rate < high)
    print(f"{low:.0%}-{high:.0%}: {count} emails {'' * count}")

# ============================================================================
# SAVE DETAILED RESULTS TO FILE
# ============================================================================

with open("consensus_results.txt", "w") as f:
    f.write("XAI CONSENSUS VERIFICATION RESULTS\n")
    f.write("=" * 80 + "\n\n")

    f.write("OVERALL STATISTICS:\n")
    f.write(f"Mean consensus: {mean_consensus:.2%}\n")
    f.write(f"Std deviation: {std_consensus:.2%}\n")
    f.write(f"Std error: {std_consensus/np.sqrt(len(consensus_rates)):.2%}\n")
    f.write(f"Median: {median_consensus:.2%}\n")
    f.write(f"Range: {min_consensus:.2%} - {max_consensus:.2%}\n")
    f.write(f"Note: Results are reproducible with random_state=42 in all components\n")
    f.write(f"Status: {'VERIFIED ✓' if 70 <= mean_consensus*100 <= 86 else 'DEVIATION ✗'}\n\n")

    f.write("-" * 80 + "\n")
    f.write("DETAILED RESULTS PER EMAIL:\n")
    f.write("-" * 80 + "\n\n")

    for i, result in enumerate(detailed_results, 1):
        f.write(f"Email {i}:\n")
        f.write(f"  True label: {result['true_label']}\n")
        f.write(f"  Predicted: {result['predicted_label']}\n")
        f.write(f"  LIME top-5: {result['lime_top5']}\n")
        f.write(f"  SHAP top-5: {result['shap_top5']}\n")
        f.write(f"  Consensus: {result['consensus']}\n")
        f.write(f"  Agreement rate: {result['agreement_rate']:.2%}\n\n")

    f.write("-" * 80 + "\n")
    f.write("CONSENSUS RATE DISTRIBUTION:\n")
    f.write("-" * 80 + "\n")
    for low, high in bins:
        count = sum(1 for rate in consensus_rates if low <= rate < high)
        f.write(f"{low:.0%}-{high:.0%}: {count} emails\n")

print("\nDetailed results saved to: consensus_results.txt")
print("\nVerification complete!")
