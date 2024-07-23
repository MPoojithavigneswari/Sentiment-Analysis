import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load the saved TF-IDF vectorizer, StandardScaler, and logistic regression model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Function to preprocess text
def preprocess_text(text, vectorizer, scaler):
    tfidf_text = vectorizer.transform([text])
    scaled_text = scaler.transform(tfidf_text)
    return scaled_text

# Streamlit app
st.title("Sentiment Analysis")
st.write("Enter a review and get its sentiment prediction.")

user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input:
        preprocessed_input = preprocess_text(user_input, tfidf_vectorizer, scaler)
        prediction = lr_model.predict(preprocessed_input)
        sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to predict.")
