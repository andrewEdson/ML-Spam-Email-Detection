# Using the Best Modle form Evaluation (SVM) to feed into ChatGPT for explanations using Lime and SHAP
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import string
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI
from lime.lime_text import LimeTextExplainer
import shap


# Load the balanced dataset
data = pd.read_csv("Data/spam_ham_dataset.csv")
data.head()

data.shape

# sns.countplot(x="label", data=data)
# plt.show()

ham_msg = data[data["label"] == "ham"]
spam_msg = data[data["label"] == "spam"]

# Downsample Ham emails to match the number of Spam emails
ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)

# Combine balanced data
balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)

# Visualize the balanced dataset
# sns.countplot(x="label", data=balanced_data)
# plt.title("Balanced Distribution of Spam and Ham Emails")
# plt.xticks(ticks=[0, 1], labels=["Ham (Not Spam)", "Spam"])
# plt.show()

balanced_data["text"] = balanced_data["text"].str.replace("Subject", "")
balanced_data.head()

punctuations_list = string.punctuation


def remove_punctuations(text):
    temp = str.maketrans("", "", punctuations_list)
    return text.translate(temp)


balanced_data["text"] = balanced_data["text"].apply(lambda x: remove_punctuations(x))
balanced_data.head()


def remove_stopwords(text):
    stop_words = stopwords.words("english")

    imp_words = []

    # Storing the important words
    for word in str(text).split():
        word = word.lower()

        if word not in stop_words:
            imp_words.append(word)

    output = " ".join(imp_words)

    return output


balanced_data["text"] = balanced_data["text"].apply(lambda text: remove_stopwords(text))
balanced_data.head()

train_X, test_X, train_Y, test_Y = train_test_split(
    balanced_data["text"], balanced_data["label"], test_size=0.2, random_state=42
)

# Load SVM model and vectorizer
svm_classifier = joblib.load("models/svm_model.pkl")
svm_vectorizer = joblib.load("models/svm_vectorizer.pkl")

# ChatGPT API Setup for explanations using Lime and SHAP would go here
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Set up Lime explainer
vectorized_train_X = svm_vectorizer.transform(train_X)
lime_explainer = LimeTextExplainer(class_names=["ham", "spam"])

# Set up SHAP explainer
shap_explainer = shap.Explainer(
    svm_classifier.predict, svm_vectorizer.transform(train_X)
)

# Get Lime explanation for a sample email
sample_email = test_X.iloc[0]
sample_email_vectorized = svm_vectorizer.transform([sample_email])


# Create a wrapper function for SVM prediction that returns probabilities
def svm_predict_proba(texts):
    """Wrapper to return decision function as probabilities for LIME"""
    vectorized = svm_vectorizer.transform(texts)
    decision = svm_classifier.decision_function(vectorized)
    # Convert decision function to probability-like scores
    import numpy as np

    proba = np.column_stack([1 - decision, decision])
    return proba


lime_exp = lime_explainer.explain_instance(
    sample_email, svm_predict_proba, num_features=10
)
print("LIME Explanation for sample email:")
print(lime_exp.as_list())

# Get SHAP-like explanation using SVM coefficients
# For linear SVM, we can use the model coefficients multiplied by feature values
import numpy as np

# Get feature names
feature_names = svm_vectorizer.get_feature_names_out()

# Get SVM coefficients (feature importance)
svm_coef = svm_classifier.coef_.toarray()[0]

# Get the sample's feature values
sample_features = sample_email_vectorized.toarray()[0]

# Calculate feature importance (coefficient * feature value)
feature_importance = svm_coef * sample_features

# Create list of (feature_name, importance) pairs for non-zero features
shap_importance = [
    (feature_names[i], feature_importance[i])
    for i in range(len(feature_names))
    if sample_features[i] != 0
]

# Sort by absolute importance and get top 10
shap_importance_sorted = sorted(shap_importance, key=lambda x: abs(x[1]), reverse=True)[
    :10
]

print("SHAP-like Explanation for sample email (Top 10 features):")
for feature, value in shap_importance_sorted:
    print(f"{feature}: {value}")

# Feed explanations to ChatGPT for further insights
lime_explanation_text = "\n".join(
    [f"{feature}: {weight}" for feature, weight in lime_exp.as_list()]
)
shap_explanation_text = "\n".join(
    [f"{feature}: {value}" for feature, value in shap_importance_sorted]
)

# Create comprehensive prompt for ChatGPT
prediction = svm_classifier.predict(sample_email_vectorized)[0]
decision_score = svm_classifier.decision_function(sample_email_vectorized)[0]

prompt = f"""You are an expert in machine learning explainability. I have a spam email detection model (SVM) that classified an email. I've analyzed this prediction using two different explainability methods: LIME and SHAP-like feature importance.

**Original Email Text:**
{sample_email}

**Model Prediction:** {prediction}
**Decision Score:** {decision_score:.4f} (Positive values indicate spam, negative indicate ham)

**LIME Explanation (Local Interpretable Model-agnostic Explanations):**
{lime_explanation_text}

**SHAP-like Explanation (Feature Importance Analysis):**
{shap_explanation_text}

Please provide a clear, comprehensive explanation of why the model made this prediction. Focus on:

1. **Key Features Both Methods Agree On**: Identify the words/features that both LIME and SHAP agree are most important for this prediction, and explain why these features are strong indicators of spam/ham.

2. **Strongest Indicators**: Highlight the top 3-5 features that have the highest impact according to both methods, explaining what makes them definitive spam or ham signals.

3. **Consensus Analysis**: Discuss the features where LIME and SHAP show strong agreement, as these are the most reliable indicators for this classification.

4. **Overall Assessment**: Based on the features both methods are certain about, provide a confident explanation of why this email was classified as spam or ham.

Please write this explanation in a way that would be understandable to someone without a machine learning background, focusing on the concrete reasons (specific words or patterns) that led to this classification."""

# Get explanation from ChatGPT
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "You are an expert in machine learning explainability and email spam detection.",
        },
        {"role": "user", "content": prompt},
    ],
    temperature=0.7,
    max_tokens=1000,
)

llm_explanation = response.choices[0].message.content
print("\n" + "=" * 80)
print("LLM EXPLANATION:")
print("=" * 80)
print(llm_explanation)
