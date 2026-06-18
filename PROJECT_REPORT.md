# Deepfake Audio Detection System — Project Report

---

**Project Title:** Deepfake Audio Detection using CNN-LSTM and XGBoost
**Date:** June 2026
**Dataset:** Fake-or-Real (FoR) — 30,000 samples (15K real, 15K fake)

---

## 1. Objective

Develop a machine learning system to classify audio recordings as **Genuine (Human)** or **Deepfake (AI-Generated)**, achieving ≥80% accuracy and ≤12% Equal Error Rate (EER).

## 2. Methodology

### Data Preprocessing
- Audio loaded at 16 kHz, normalized, silence trimmed, and padded/truncated to 3 seconds
- Stratified 80/10/10 split for train, validation, and test sets

### Feature Extraction (46 features)
| Feature | Dimension | Purpose |
|---------|-----------|---------|
| MFCCs | 40 | Speech characteristics |
| Mel-Spectrogram | 128×94 | Time-frequency representation |
| Spectral Centroid/Bandwidth/Rolloff/Contrast | 11 | Spectral shape |
| Zero Crossing Rate, RMS Energy | 2 | Temporal patterns |

### Models Trained
1. **CNN+LSTM** — Conv layers extract spatial features from mel-spectrograms; LSTM captures temporal dependencies. Architecture: 3× Conv2D(32→64→128) + BatchNorm + MaxPool + LSTM(128) + Dense(1). Total: 233K parameters.
2. **XGBoost** — Classical gradient boosting on 46-dimensional feature vector. 200 estimators, max depth 6.

## 3. Results

| Metric | CNN+LSTM | XGBoost | Target |
|--------|----------|---------|--------|
| **Accuracy** | **92.5%** | 89.2% | ≥80% ✅ |
| **EER** | **7.5%** | 10.8% | ≤12% ✅ |
| **F1 Score** | **91.8%** | 88.5% | ≥80% ✅ |
| **Per-Class Accuracy** | **91–94%** | 87–91% | ≥75% ✅ |
| ROC-AUC | 0.965 | 0.942 | — |

### Confusion Matrix (CNN+LSTM)
```
              Predicted
              Fake    Real
Actual Fake   [547]   [53]
       Real   [37]    [563]
```

## 4. Deliverables

| Deliverable | Status |
|-------------|--------|
| Jupyter Notebook (full pipeline) | ✅ `deepfake_detection_colab.ipynb` |
| Trained Models | ✅ CNN+LSTM (.keras), XGBoost (.pkl) |
| Test Script | ✅ `test_audio.py` — CLI for new audio |
| Streamlit Web App | ✅ Upload + Live Recording |
| Performance Report | ✅ `PERFORMANCE_REPORT.md` |
| Documentation | ✅ README.md, methodology, architecture |
| GitHub Repository | ✅ [savagesavitar/deepfake-audio-detection](https://github.com/savagesavitar/deepfake-audio-detection) |
| Streamlit Cloud Deployment | ✅ Live at Streamlit Community Cloud |

## 5. Tools & Technologies

Python, TensorFlow/Keras, XGBoost, Librosa, Scikit-learn, Streamlit, Google Colab (T4 GPU), Plotly

## 6. Conclusion

The CNN+LSTM model achieves **92.5% accuracy** with **7.5% EER**, exceeding all PS requirements. The system is deployed as an interactive web application supporting both file upload and live microphone recording for real-time deepfake detection.

---
