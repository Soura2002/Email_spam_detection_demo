import streamlit as st
import pandas as pd
import numpy as np
import string
from collections import Counter
from nltk.corpus import stopwords
import nltk
import pickle

# Download NLTK stopwords (run once)
nltk.download('stopwords')

# Load stopwords and punctuation
stop_words = set(stopwords.words("english"))
important_chars = {'!', '$'}
custom_punctuation = set(string.punctuation) - important_chars

# Load trained model
model = pickle.load(open('email_spam_detection_model.pkl','rb'))  # Ensure this file exists

# Feature names (must match model training)
feature_names = ['word_freq_make',
 'word_freq_remove',
 'word_freq_internet',
 'word_freq_order',
 'word_freq_receive',
 'word_freq_people',
 'word_freq_report',
 'word_freq_addresses',
 'word_freq_free',
 'word_freq_business',
 'word_freq_credit',
 'word_freq_000',
 'word_freq_money',
 'word_freq_hp',
 'word_freq_hpl',
 'word_freq_george',
 'word_freq_lab',
 'word_freq_labs',
 'word_freq_telnet',
 'word_freq_857',
 'word_freq_data',
 'word_freq_415',
 'word_freq_85',
 'word_freq_pm',
 'word_freq_cs',
 'word_freq_meeting',
 'word_freq_original',
 'word_freq_project',
 'word_freq_edu',
 'word_freq_conference',
 'char_freq_!',
 'char_freq_$',
 'cap_run_length_total_bin',
 'capital_run_length_average_bin',
 'capital_run_length_longest_bin']

# ---------------- Preprocessing Function ----------------
def preprocess_input(email_text):
    original_text = email_text
    text = email_text.lower()

    # Remove punctuation (except ! and $)
    cleaned_text = ''.join(ch for ch in text if ch not in custom_punctuation)
    words = [word for word in cleaned_text.split() if word not in stop_words]
    word_counts = Counter(words)
    total_words = len(words) if len(words) > 0 else 1

    features = {}

    # Word frequency
    for col in feature_names:
        if col.startswith("word_freq_"):
            word = col.replace("word_freq_", "")
            features[col] = (word_counts[word] / total_words) * 100 if word in word_counts else 0

    # Char frequency
    for col in feature_names:
        if col.startswith("char_freq_"):
            char = col.replace("char_freq_", "")
            features[col] = (text.count(char) / len(text)) * 100 if len(text) > 0 else 0

    # Capital run features (original case)
    caps = [len(c) for c in ''.join([ch if ch.isupper() else ' ' for ch in original_text]).split()]
    cap_avg = np.mean(caps) if caps else 0
    cap_long = max(caps) if caps else 0
    cap_total = sum(caps)

    
    # Binary features
    features["cap_run_length_total_bin"] = int(cap_total > 225)
    features["capital_run_length_average_bin"] = int(cap_avg > 4)
    features["capital_run_length_longest_bin"] = int(cap_long > 50)

    # Return dataframe in model-compatible format
    return pd.DataFrame([features])[feature_names]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📧 Spam Email Classifier", layout="centered")
st.title("📧 Spam Classifier with Preprocessing")
st.markdown("Paste an email below. The system will preprocess the text and predict if it's **spam or not**.")

email_input = st.text_area("📨 Enter email text here:", height=200)

if st.button("🔍 Analyze Email"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter an email message.")
    else:
        processed = preprocess_input(email_input)
        prediction = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 **Spam** detected with {prob*100:.2f}% confidence.")
        else:
            st.success(f"✅ **Not Spam** with {prob*100:.2f}% confidence.")