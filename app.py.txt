import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App Title
st.title("📰 Fake News Detector")
st.subheader("Check if a news article is FAKE or REAL")

# Input Text
news_text = st.text_area("Enter the news article content below:")

# Predict button
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Transform and predict
        tfidf = vectorizer.transform([news_text])
        prediction = model.predict(tfidf)[0]

        # Show result
        if prediction == 0:
            st.error("❌ This news is likely FAKE.")
        else:
            st.success("✅ This news is likely REAL.")
