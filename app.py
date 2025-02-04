import streamlit as st
import joblib
import numpy as np

# Load the saved model and TF-IDF vectorizer
@st.cache_resource  # Cache the model for faster loading
def load_model():
    model = joblib.load('quora_duplicate_rf_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_model()

# Streamlit app
st.title("Quora Duplicate Question Classifier")
st.write("Enter two questions to check if they are duplicates.")

# Input fields for questions
question1 = st.text_input("Enter Question 1:")
question2 = st.text_input("Enter Question 2:")

# Predict button
if st.button("Check Duplicate"):
    if question1 and question2:
        # Preprocess the input questions
        q1_tfidf = tfidf.transform([question1])
        q2_tfidf = tfidf.transform([question2])
        X = np.hstack((q1_tfidf.toarray(), q2_tfidf.toarray()))

        # Make prediction
        prediction = model.predict(X)
        result = "Duplicate" if prediction[0] == 1 else "Not Duplicate"

        # Display the result
        st.success(f"The questions are: **{result}**")
    else:
        st.error("Please enter both questions.")