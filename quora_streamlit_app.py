import streamlit as st
import re
import nltk
import distance
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

# Download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Function to preprocess text
def clean_text(q):
    q = str(q).lower().strip()
    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'\W', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q

# Function to get common words
def common_words(q1, q2):
    return len(set(q1.split()) & set(q2.split()))

# Function to get total words
def total_words(q1, q2):
    return (len(q1.split()) + len(q2.split())) / 2

# Function to get fuzzy similarity scores
def fuzzy_features(q1, q2):
    return {
        "Token Sort Ratio": fuzz.token_sort_ratio(q1, q2),
        "Token Set Ratio": fuzz.token_set_ratio(q1, q2),
        "Partial Ratio": fuzz.partial_ratio(q1, q2)
    }

# Streamlit UI
st.title("Duplicate Question Detection")
st.write("Enter two questions to check their similarity.")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check Similarity"):
    if q1 and q2:
        # Clean and preprocess
        q1_clean = clean_text(q1)
        q2_clean = clean_text(q2)

        # Extract features
        common = common_words(q1_clean, q2_clean)
        total = total_words(q1_clean, q2_clean)
        fuzz_scores = fuzzy_features(q1_clean, q2_clean)

        # Display results
        st.subheader("Similarity Features")
        st.write(f"🔹 **Common Words:** {common}")
        st.write(f"🔹 **Average Total Words:** {total}")

        st.subheader("Fuzzy Matching Scores")
        st.write(f"🔹 **Token Sort Ratio:** {fuzz_scores['Token Sort Ratio']}")
        st.write(f"🔹 **Token Set Ratio:** {fuzz_scores['Token Set Ratio']}")
        st.write(f"🔹 **Partial Ratio:** {fuzz_scores['Partial Ratio']}")

        # Suggest if they might be duplicates
        avg_similarity = np.mean(list(fuzz_scores.values()))
        if avg_similarity > 75:
            st.success("✅ The questions are likely duplicates!")
        else:
            st.warning("❌ The questions are probably different.")
    else:
        st.error("Please enter both questions!")

