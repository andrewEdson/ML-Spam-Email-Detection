import os
import glob
import string
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import seaborn as sns
from nltk.corpus import stopwords

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

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
train_X_vectors = vectorizer.fit_transform(train_X)
test_X_vectors = vectorizer.transform(test_X)

lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(train_X_vectors, train_Y)
predictions = lr_classifier.predict(test_X_vectors)
print(classification_report(test_Y, predictions))
