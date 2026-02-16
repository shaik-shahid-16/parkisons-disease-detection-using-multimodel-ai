import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import librosa
import parselmouth
from PIL import Image
import io

# App Settings
st.set_page_config(
    page_title="Multimodal Parkinson's Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    .big-font {
        font-size:32px !important;
        font-weight: bold;
        color: #073763;
        }
    .result-box {
        background: linear-gradient(90deg, #d1eaff 0%, #b6dfdb 100%);
        padding: 1.2em 2em;
        border-radius: 18px;
        margin-top: 1em;
        margin-bottom: 1em;
        font-size: 22px;
        }
    .confidence-bar-container {
        background-color: #ebf7fa;
        border-radius: 100px;
        height: 22px;
        margin-bottom:20px;
        margin-top:10px;
        }
    .confidence-bar {
        height: 100%;
        border-radius: 100px;
        background-image: linear-gradient(to right, #00b4d8 , #4fffb0 80%);
    }
    .footer {font-size:13px; margin-top:2em;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-font">🧠 Multimodal Parkinson\'s Disease Detector</div>', unsafe_allow_html=True)
st.write("This app fuses handwriting & voice biomarker AI for robust Parkinson’s diagnosis. Upload a handwriting image and a short voice clip (sustained vowel recommended) to get a fused AI result with confidence.")

# Load models, only once on app start
@st.cache_resource
def load_models():
    handwriting_model = load_model("handwriting_parkinsons_96acc.keras")
    voice_model = joblib.load("voice_ensemble_model.joblib")
    voice_scaler = joblib.load("voice_scaler.joblib")
    return handwriting_model, voice_model, voice_scaler

handwriting_model, voice_model, voice_scaler = load_models()
class_names = ['Healthy', 'Parkinsons']

def extract_parkinson_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_stack = np.concatenate([mfcc, delta, delta2], axis=0)
    mfcc_mean = np.mean(mfcc_stack, axis=1)
    mfcc_std = np.std(mfcc_stack, axis=1)
    try:
        snd = parselmouth.Sound(file_path)
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = parselmouth.praat.call([snd, pp], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = snd.to_harmonicity_ac().values.mean()
    except Exception:
        jitter = shimmer = hnr = 0.0
    return np.concatenate([mfcc_mean, mfcc_std, [jitter, shimmer, hnr]])

st.markdown("### 📄 Step 1: Upload Handwriting Image")
img_file = st.file_uploader("Choose a handwriting sample (png/jpg)", type=["png", "jpg", "jpeg"], key="hw_upload")
img_array = None

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Handwriting Sample", use_column_width=True)
    IMG_SIZE = 256
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

st.markdown("### 🎤 Step 2: Upload Voice Sample (.wav)")
voice_file = st.file_uploader("Choose a short voice recording (.wav)", type=["wav"], key="voice_upload")
voice_probs = None

if st.button("🔎 Analyze both and Predict Diagnosis", use_container_width=True):
    if img_array is None or voice_file is None:
        st.warning("Please upload both a handwriting image and a voice sample.")
    else:
        with st.spinner("Analyzing handwriting..."):
            hand_probs = handwriting_model.predict(img_array)[0]
        with st.spinner("Analyzing voice biomarker..."):
            # Save .wav to disk for feature extraction
            temp_voice_path = "temp_voice.wav"
            with open(temp_voice_path, "wb") as f:
                f.write(voice_file.read())
            voice_feats = extract_parkinson_features(temp_voice_path).reshape(1, -1)
            voice_feats_scaled = voice_scaler.transform(voice_feats)
            voice_probs = voice_model.predict_proba(voice_feats_scaled)[0]

        # Weighted fusion for best clinical interpretability
        hand_weight, voice_weight = 0.65, 0.35
        fused_probs = hand_weight * hand_probs + voice_weight * voice_probs
        fused_pred_idx = int(np.argmax(fused_probs))
        fused_confidence = fused_probs[fused_pred_idx]

        st.markdown(f'<div class="result-box">FINAL Multimodal Diagnosis: <b>{class_names[fused_pred_idx]}</b><br>Confidence: <b>{fused_confidence*100:.1f}%</b></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <b>Handwriting Confidence:</b>
        <div class="confidence-bar-container">
            <div class="confidence-bar" style="width:{hand_probs[np.argmax(hand_probs)]*100:.1f}%"></div>
        </div>
        <b>Voice Confidence:</b>
        <div class="confidence-bar-container">
            <div class="confidence-bar" style="width:{voice_probs[np.argmax(voice_probs)]*100:.1f}%"></div>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"Handwriting model: {class_names[np.argmax(hand_probs)]} ({hand_probs[np.argmax(hand_probs)]*100:.1f}%)")
        st.write(f"Voice model: {class_names[np.argmax(voice_probs)]} ({voice_probs[np.argmax(voice_probs)]*100:.1f}%)")
        st.success("Analysis complete! See breakdown above.")

st.markdown('<div class="footer">Copyright © 2025. Built with Keras, scikit-learn, librosa, and Parselmouth. The tool is experimental and for educational/triage use only—it does not provide a medical diagnosis.</div>', unsafe_allow_html=True)
