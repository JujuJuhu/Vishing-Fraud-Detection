import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vishing Detector", page_icon="📞")

# 1. Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('vishing_model.pkl')

try:
    model = load_model()
except:
    st.error("Model file not found! Please run train.py first to create 'vishing_model.pkl'.")
    st.stop()

# 2. UI Setup
st.title("🛡️ Voice Phishing (Vishing) Detection")
uploaded_file = st.file_uploader("Upload an audio file (.mp3, .wav)", type=["mp3", "wav"])

if uploaded_file:
    # Process Audio
    with st.spinner("Analyzing..."):
        y, sr = librosa.load(uploaded_file, sr=16000)
        
        # Feature Extraction (Same as training)
        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)
        mfcc_norm = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)
        features = np.concatenate([np.mean(mfcc_norm, axis=1), np.std(mfcc_norm, axis=1)]).reshape(1, -1)
        
        # Prediction
        prediction = model.predict(features)
        prob = model.predict_proba(features)[0]

    # Display Results
    st.divider()
    if prediction[0] == 0:
        st.error(f"🚨 **RESULT: VISHING (FRAUD) DETECTED** (Confidence: {prob[0]:.2%})")
    else:
        st.success(f"✅ **RESULT: NORMAL CALL** (Confidence: {prob[1]:.2%})")

    # Visuals
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=16000, ax=ax)
    st.pyplot(fig)
    st.audio(uploaded_file)