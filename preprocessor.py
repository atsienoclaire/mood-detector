# ===== IMPORT LIBRARIES =====
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ===== DOWNLOAD STOPWORDS (if not already downloaded) =====
nltk.download('stopwords')

# ===== LOAD DATA =====
df = pd.read_csv('data/train.txt', sep=';', header=None, names=['text', 'emotion'])

# ===== CLEANING =====
# 1. Remove missing and duplicate rows
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# 2. Clean text: lowercase, remove special characters
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove anything not a-z or space
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# 3. Remove stopwords
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

# ===== ENCODING =====
# Encode emotion labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['emotion'])

# ===== SPLIT DATA =====
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data cleaning complete.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ===== FEATURE ENGINEERING: TF-IDF =====
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("TF-IDF feature matrix shape (training):", X_train_vec.shape)
print("TF-IDF feature matrix shape (testing):", X_test_vec.shape)
print("✅ Feature engineering complete.")
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ====== 1. NAIVE BAYES MODEL ======
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

print("\n🔍 Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print("\nClassification Report:\n", classification_report(y_test, nb_preds, target_names=label_encoder.classes_))

# Save model
joblib.dump(nb_model, "naive_bayes_model.pkl")

# ====== 2. LOGISTIC REGRESSION ======
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train_vec, y_train)
lr_preds = lr_model.predict(X_test_vec)

print("\n🔍 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print("\nClassification Report:\n", classification_report(y_test, lr_preds, target_names=label_encoder.classes_))

# Save model
joblib.dump(lr_model, "logistic_regression_model.pkl")

# ========== CONFUSION MATRIX FOR NAIVE BAYES ==========
nb_cm = confusion_matrix(y_test, nb_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ========== CONFUSION MATRIX FOR LOGISTIC REGRESSION ==========
lr_cm = confusion_matrix(y_test, lr_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
# Function to calculate all metrics
def model_metrics(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred, average='weighted'), 3),
        "Recall": round(recall_score(y_true, y_pred, average='weighted'), 3),
        "F1-Score": round(f1_score(y_true, y_pred, average='weighted'), 3)
    }

# Calculate for both models
nb_results = model_metrics(y_test, nb_preds)
lr_results = model_metrics(y_test, lr_preds)

# Display comparison table
comparison_df = pd.DataFrame([nb_results, lr_results],
                             index=["Naive Bayes", "Logistic Regression"])

print("\n📊 Model Performance Comparison:")
print(comparison_df)



# ===== EDA (Exploratory Data Analysis) =====
sns.set(style="whitegrid")

# 1. Distribution of Emotions
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='emotion', order=df['emotion'].value_counts().index, palette='Set2')
plt.title("Distribution of Emotions")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Text Length Distribution
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(df['text_length'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Text Lengths")
plt.xlabel("Text Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3. Boxplot of Text Length by Emotion
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='emotion', y='text_length', palette='pastel')
plt.title("Text Lengths by Emotion")
plt.xlabel("Emotion")
plt.ylabel("Text Length")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("✅ EDA complete.")

# ========== INTERPRETING FEATURES FROM LOGISTIC REGRESSION ==========
feature_names = vectorizer.get_feature_names_out()
coefs = lr_model.coef_

print("\n🔍 Top Influential Words per Emotion Class:\n")

# Show top 10 positive contributing words per emotion
for i, emotion in enumerate(label_encoder.classes_):
    top10 = np.argsort(coefs[i])[-10:]  # Top 10 coefficients
    print(f"{emotion.capitalize()}: {[feature_names[j] for j in top10][::-1]}")  # reversed for descending

joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

import joblib  # If not already imported

# Save model, vectorizer, and label encoder
joblib.dump(lr_model, "logistic_regression_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model, vectorizer, and label encoder saved.")






