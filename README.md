# 🧠 Parkinson’s Disease Multimodal Detection AI

Try it now: [🚀 Launch the Multimodal Parkinson’s Detector](https://parkinsonsmultimodel-9o3tbugc3pleuxxglhhgwa.streamlit.app/)

---

## Overview

This repository implements a state-of-the-art multimodal AI system for Parkinson’s disease detection, combining:

- **Handwriting Image Deep Learning (EfficientNet)**
- **Voice Signal Analysis (ensemble model on extracted pathology features)**
- **Weighted fusion** for a single, robust diagnostic prediction

These models are packaged in an attractive and easy-to-use [Streamlit](https://streamlit.io/) web interface where users upload a handwriting sample and a short voice file, then instantly receive a fused diagnosis and confidence metrics.

---

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parkinsonsmultimodel-9o3tbugc3pleuxxglhhgwa.streamlit.app/)

- **Test it in your browser here:**  
  https://parkinsonsmultimodel-9o3tbugc3pleuxxglhhgwa.streamlit.app/

---

## Features

- End-to-end fusion of handwriting and voice biomarkers for highly accurate detection
- Advanced voice features (MFCC+delta, jitter, shimmer, HNR)
- Robust ensemble classifier (RandomForest, KNN, MLP) for voice
- Fine-tuned EfficientNet model for handwriting images
- Intuitive, step-by-step Streamlit app with clear confidence output
- Cross-validated accuracy:  
  - Handwriting: ~96%  
  - Voice (augmented): >85%  
  - Fused: Highly robust, interpretable final probabilities

---

## Folder Structure

ParkinsonsMultimodel/
├── app.py
├── handwriting_parkinsons_96acc.keras
├── voice_ensemble_model.joblib
├── voice_scaler.joblib
├── requirements.txt
├── README.md
└── .streamlit/
└── runtime.txt


---

## Getting Started

### 1. Clone This Repository

git clone https://github.com/saahith-k/ParkinsonsMultimodel.git
cd ParkinsonsMultimodel


### 2. Install Requirements

pip install -r requirements.txt


### 3. Run Locally

streamlit run app.py


Visit `http://localhost:8501` to use the app.

---

## Usage

- **Step 1:** Upload a handwriting image sample (.jpg or .png).
- **Step 2:** Upload a short voice sample (.wav, mono recommended).
- **Step 3:** Click “Analyze both and Predict Diagnosis”.
- **Step 4:** See the fused AI diagnosis and confidence for each modality.

---

## Model Details

**Handwriting Model**  
- EfficientNetB0, input: 256x256 color images  
- Trained on handwritten samples for Parkinson’s/Healthy classification

**Voice Model**  
- Features: MFCCs (40), deltas, jitter, shimmer, HNR  
- Ensemble: RandomForest, KNN, MLP  
- Data augmentation, SMOTE class balancing

**Fusion**  
- Weighted averaging of probabilities (handwriting: 65%, voice: 35%)
- Transparent per-modality and overall result display

---

## Live App

Click here to try the app:  
👉 [https://parkinsonsmultimodel-9o3tbugc3pleuxxglhhgwa.streamlit.app/](https://parkinsonsmultimodel-9o3tbugc3pleuxxglhhgwa.streamlit.app/)

---

## Disclaimer

This tool is for research, demonstration, and educational purposes only.  
**It does not provide a medical diagnosis.**  
Always consult a medical professional for clinical decisions.

---

## License

[MIT License](LICENSE) — see LICENSE file for details.

---


