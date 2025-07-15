import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# === Load models ===
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Clean user input ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Streamlit UI ===
st.title("🧠 Mood Detector from Text")
st.markdown("Enter a sentence below to detect its emotional tone:")

user_input = st.text_area("Text Input", "")

if st.button("Predict Mood"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        emotion = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🎉 Predicted Mood: **{emotion.upper()}**")
