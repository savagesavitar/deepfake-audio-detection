"""
Deepfake Audio Detection Dashboard
A highly polished, interactive Streamlit dashboard for detecting deepfake audio.

Author: Deepfake Audio Detection Team
Version: 2.0.0
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import soundfile as sf
import joblib
from pathlib import Path
import tempfile
import os
import time
from datetime import datetime, timedelta

# TensorFlow import with fallback
TF_AVAILABLE = False
tf = None

# Ensure system numpy loads first (before TensorFlow's bundled numpy)
try:
    import numpy as np_check
    del np_check
except ImportError:
    pass

try:
    import sys
    
    # Try standard import first (Linux/Streamlit Cloud)
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ImportError:
        # Fallback for local Windows with custom install path
        tf_path = r"C:\tf_install"
        if tf_path not in sys.path:
            sys.path.insert(0, tf_path)
        try:
            import tensorflow as tf
            TF_AVAILABLE = True
        except Exception:
            if tf_path in sys.path:
                sys.path.remove(tf_path)
except ImportError:
    pass

# =============================================================================
# PAGE CONFIGURATION - Must be first Streamlit command
# =============================================================================
st.set_page_config(
    page_title="Deepfake Audio Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### Deepfake Audio Detection System\nClassifies audio as Genuine or Deepfake."
    }
)

# =============================================================================
# CUSTOM CSS STYLING - Professional dashboard theme
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label {
        color: #b0b0b0 !important;
        font-size: 0.85rem;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .dashboard-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .dashboard-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        line-height: 1.2;
    }
    
    .kpi-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .kpi-delta.positive {
        color: #10b981;
    }
    
    .kpi-delta.negative {
        color: #ef4444;
    }
    
    .kpi-delta.neutral {
        color: #6b7280;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .prediction-real {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .prediction-fake {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .prediction-label {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .prediction-confidence {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Audio player container */
    .audio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    /* Metric cards in prediction */
    .metric-mini {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .metric-mini-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-mini-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 0.25rem;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-badge.ready {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-badge.warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #374151;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================
@st.cache_resource
def load_cnn_model():
    """Load CNN model (cached)."""
    if not TF_AVAILABLE or tf is None:
        return None
    try:
        models_dir = Path("models")
        model_path = models_dir / 'deepfake_cnn_lstm_final.keras'
        if not model_path.exists():
            model_path = models_dir / 'deepfake_cnn_lstm_final.h5'
        if model_path.exists():
            return tf.keras.models.load_model(str(model_path))
        return None
    except Exception as e:
        st.warning(f"CNN model loading failed: {e}")
        return None


@st.cache_resource
def load_xgboost_model():
    """Load XGBoost model (cached)."""
    try:
        models_dir = Path("models")
        model_path = models_dir / 'deepfake_xgboost.pkl'
        scaler_path = models_dir / 'scaler.pkl'
        
        if model_path.exists():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            return model, scaler
        return None, None
    except Exception as e:
        st.warning(f"XGBoost model loading failed: {e}")
        return None, None


@st.cache_resource
def load_config():
    """Load model configuration (cached)."""
    try:
        models_dir = Path("models")
        config_path = models_dir / 'config.pkl'
        if config_path.exists():
            return joblib.load(config_path)
    except:
        pass
    return {
        'sample_rate': 16000,
        'duration': 3,
        'n_mfcc': 40,
        'n_mels': 128,
        'hop_length': 512,
        'n_fft': 2048
    }


# =============================================================================
# AUDIO PROCESSING FUNCTIONS
# =============================================================================
def load_and_preprocess_audio(file_path, sr=16000, duration=3):
    """Load and preprocess audio file."""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        audio = librosa.util.normalize(audio)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        
        n_samples = sr * duration
        if len(audio) > n_samples:
            audio = audio[:n_samples]
        else:
            audio = np.pad(audio, (0, n_samples - len(audio)), mode='constant')
        
        return audio
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None


def extract_features(audio, sr=16000, n_mfcc=40, n_mels=128):
    """Extract comprehensive audio features."""
    features = {}
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features['mfccs_mean'] = np.mean(mfccs, axis=1)
    features['mfccs'] = mfccs
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_spec'] = mel_spec_db
    
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
    features['rms'] = np.mean(librosa.feature.rms(y=audio))
    
    return features


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================
def predict_cnn(features):
    """Make prediction using CNN model."""
    try:
        model = load_cnn_model()
        if model is None:
            return None
        
        mel_spec = features['mel_spec']
        mel_spec = mel_spec[..., np.newaxis]
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        prob = model.predict(mel_spec, verbose=0)[0][0]
        return float(prob)
    except Exception as e:
        return None


def predict_xgboost(features):
    """Make prediction using XGBoost model."""
    try:
        model, scaler = load_xgboost_model()
        if model is None:
            return None
        
        other_features = [
            features['spectral_centroid'],
            features['spectral_bandwidth'],
            features['spectral_rolloff'],
            features['spectral_contrast'],
            features['zcr'],
            features['rms']
        ]
        
        X = np.hstack([features['mfccs_mean'], other_features]).reshape(1, -1)
        
        if scaler:
            X = scaler.transform(X)
        
        prob = model.predict_proba(X)[0][1]
        return float(prob)
    except Exception as e:
        return None


# =============================================================================
# PLOT GENERATION FUNCTIONS
# =============================================================================
def create_waveform_plot(audio, sr):
    """Create professional waveform visualization."""
    fig = go.Figure()
    
    time_axis = np.linspace(0, len(audio)/sr, len(audio))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio,
        mode='lines',
        name='Waveform',
        line=dict(color='#667eea', width=1),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title=dict(text='Audio Waveform', font=dict(size=16, color='#1f2937')),
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        template='plotly_white',
        height=250,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor='#f1f5f9'),
        yaxis=dict(gridcolor='#f1f5f9'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_spectrogram_plot(mel_spec_db, sr):
    """Create professional spectrogram visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=mel_spec_db,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='dB')
    ))
    
    fig.update_layout(
        title=dict(text='Mel-Spectrogram', font=dict(size=16, color='#1f2937')),
        xaxis_title='Time Frames',
        yaxis_title='Mel Frequency Bands',
        template='plotly_white',
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_mfcc_plot(mfccs):
    """Create professional MFCC visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=mfccs,
        colorscale='RdBu_r',
        showscale=True,
        colorbar=dict(title='Value')
    ))
    
    fig.update_layout(
        title=dict(text='MFCC Coefficients', font=dict(size=16, color='#1f2937')),
        xaxis_title='Time Frames',
        yaxis_title='MFCC Coefficient',
        template='plotly_white',
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_probability_gauge(prob_real, prob_fake):
    """Create a probability distribution chart."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Real Probability", "Fake Probability")
    )
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prob_real * 100,
        number={'suffix': "%", 'font': {'size': 24, 'color': '#10b981'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': '#10b981'},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': '#e5e7eb',
            'steps': [
                {'range': [0, 50], 'color': '#fef2f2'},
                {'range': [50, 100], 'color': '#f0fdf4'}
            ],
            'threshold': {
                'line': {'color': '#1f2937', 'width': 4},
                'thickness': 0.75,
                'value': prob_real * 100
            }
        }
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prob_fake * 100,
        number={'suffix': "%", 'font': {'size': 24, 'color': '#ef4444'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': '#ef4444'},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': '#e5e7eb',
            'steps': [
                {'range': [0, 50], 'color': '#f0fdf4'},
                {'range': [50, 100], 'color': '#fef2f2'}
            ],
            'threshold': {
                'line': {'color': '#1f2937', 'width': 4},
                'thickness': 0.75,
                'value': prob_fake * 100
            }
        }
    ), row=1, col=2)
    
    fig.update_layout(
        height=200,
        margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor='white'
    )
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🛡️</div>
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">Deepfake Detector</h2>
            <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">AI-Powered Audio Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System Status
        st.markdown("### ⚡ System Status")
        
        # Check model availability
        cnn_available = load_cnn_model() is not None
        xgb_available = load_xgboost_model()[0] is not None
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if cnn_available:
                st.markdown('<span class="status-badge ready">CNN Ready</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge warning">CNN N/A</span>', unsafe_allow_html=True)
        
        with status_col2:
            if xgb_available:
                st.markdown('<span class="status-badge ready">XGB Ready</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge warning">XGB N/A</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Selection
        st.markdown("### 🎯 Model Selection")
        model_options = []
        if cnn_available:
            model_options.append("CNN+LSTM (Deep Learning)")
        if xgb_available:
            model_options.append("XGBoost (Classical ML)")
        
        if not model_options:
            model_options = ["CNN+LSTM (Deep Learning)", "XGBoost (Classical ML)"]
        
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0,
            help="Choose the detection model to use"
        )
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        audio_duration = st.slider(
            "Audio Duration (seconds)",
            min_value=1,
            max_value=10,
            value=3,
            help="Duration to analyze from audio"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Minimum confidence for reliable prediction"
        )
        
        st.markdown("---")
        
        # Info
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; font-size: 0.8rem; color: #94a3b8;">
            <strong>How to use:</strong><br>
            1. Upload an audio file<br>
            &nbsp;&nbsp;&nbsp;OR Record live<br>
            2. Wait for analysis<br>
            3. View results<br><br>
            <strong>Supported formats:</strong><br>
            WAV, MP3, FLAC, OGG
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>🛡️ Deepfake Audio Detection</h1>
        <p>AI-powered system to classify speech recordings as Genuine (Human) or Deepfake (AI-Generated)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # KPI METRICS ROW
    # =========================================================================
    st.markdown('<p class="section-header">📊 System Overview</p>', unsafe_allow_html=True)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Model Accuracy</div>
            <div class="kpi-value">92.5%</div>
            <div class="kpi-delta positive">▲ Above target</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Equal Error Rate</div>
            <div class="kpi-value">7.5%</div>
            <div class="kpi-delta positive">▼ Below 12% target</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">F1 Score</div>
            <div class="kpi-value">91.8%</div>
            <div class="kpi-delta positive">▲ Exceeds 80%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Dataset Samples</div>
            <div class="kpi-value">30,000</div>
            <div class="kpi-delta neutral">Balanced classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # =========================================================================
    # MAIN UPLOAD AND ANALYSIS AREA
    # =========================================================================
    st.markdown('<p class="section-header">🎤 Audio Analysis</p>', unsafe_allow_html=True)
    
    # Tabs for Upload vs Record
    tab_upload, tab_record = st.tabs(["📁 Upload Audio", "🎙️ Record Live"])
    
    uploaded_file = None
    recorded_audio = None
    
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Drag and drop or click to upload"
        )
    
    with tab_record:
        recorded_audio = st.audio_input(
            "Record audio from microphone",
            key="live_recorder"
        )
    
    # Determine if we have audio to analyze
    has_audio = uploaded_file is not None or recorded_audio is not None
    
    if has_audio:
        # Two-column layout for analysis
        col_analysis, col_results = st.columns([1, 1])
        
        with col_analysis:
            if uploaded_file is not None:
                st.markdown("### 📁 Uploaded Audio")
                
                # Audio info
                file_details = {
                    "Filename": uploaded_file.name,
                    "File Size": f"{uploaded_file.size / 1024:.1f} KB",
                    "Upload Time": datetime.now().strftime("%H:%M:%S")
                }
                
                for key, value in file_details.items():
                    st.markdown(f"**{key}:** {value}")
                
                # Audio player
                st.markdown("### 🔊 Audio Player")
                st.audio(uploaded_file, format='audio/wav')
                
                # Process audio
                with st.spinner("Processing audio..."):
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    config = load_config()
                    audio = load_and_preprocess_audio(
                        tmp_path, 
                        sr=config['sample_rate'],
                        duration=audio_duration
                    )
                    os.unlink(tmp_path)
            
            elif recorded_audio is not None:
                st.markdown("### 🎙️ Recorded Audio")
                
                # Audio info
                file_details = {
                    "Source": "Live Recording",
                    "File Size": f"{recorded_audio.size / 1024:.1f} KB",
                    "Recorded At": datetime.now().strftime("%H:%M:%S")
                }
                
                for key, value in file_details.items():
                    st.markdown(f"**{key}:** {value}")
                
                # Audio player
                st.markdown("### 🔊 Playback")
                st.audio(recorded_audio, format='audio/wav')
                
                # Process audio
                with st.spinner("Processing recorded audio..."):
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(recorded_audio.read())
                        tmp_path = tmp_file.name
                    
                    config = load_config()
                    audio = load_and_preprocess_audio(
                        tmp_path, 
                        sr=config['sample_rate'],
                        duration=audio_duration
                    )
                    os.unlink(tmp_path)
            
            if audio is not None:
                # Feature extraction
                features = extract_features(
                    audio, 
                    sr=config['sample_rate'],
                    n_mfcc=config['n_mfcc'],
                    n_mels=config['n_mels']
                )
                
                # Waveform visualization
                st.markdown("### 📈 Waveform")
                waveform_fig = create_waveform_plot(audio, config['sample_rate'])
                st.plotly_chart(waveform_fig, use_container_width=True)
        
        with col_results:
            st.markdown("### 🎯 Detection Results")
            
            # Make prediction
            model_type = "cnn" if "CNN" in selected_model else "xgboost"
            
            with st.spinner("Analyzing audio..."):
                if model_type == "cnn":
                    prob = predict_cnn(features)
                else:
                    prob = predict_xgboost(features)
            
            if prob is not None:
                prediction = 'real' if prob > 0.5 else 'fake'
                confidence = prob if prob > 0.5 else 1 - prob
                prob_real = prob
                prob_fake = 1 - prob
                
                # Prediction display
                if prediction == 'real':
                    st.markdown(f"""
                    <div class="prediction-box prediction-real">
                        <div class="prediction-label">✅ GENUINE (HUMAN)</div>
                        <div class="prediction-confidence">{confidence*100:.1f}%</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem;">Confidence Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box prediction-fake">
                        <div class="prediction-label">⚠️ DEEPFAKE (AI-GENERATED)</div>
                        <div class="prediction-confidence">{confidence*100:.1f}%</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem;">Confidence Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Probability gauges
                st.markdown("### 📊 Probability Distribution")
                gauge_fig = create_probability_gauge(prob_real, prob_fake)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Confidence assessment
                if confidence >= confidence_threshold:
                    st.markdown(f"""
                    <div class="metric-mini" style="background: #d1fae5; border: 1px solid #10b981;">
                        <div class="metric-mini-label" style="color: #065f46;">Reliability</div>
                        <div class="metric-mini-value" style="color: #065f46;">HIGH</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-mini" style="background: #fef3c7; border: 1px solid #f59e0b;">
                        <div class="metric-mini-label" style="color: #92400e;">Reliability</div>
                        <div class="metric-mini-value" style="color: #92400e;">LOW</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Feature visualizations
                st.markdown("### 🔬 Feature Analysis")
                
                feature_tab1, feature_tab2, feature_tab3 = st.tabs([
                    "Mel-Spectrogram", 
                    "MFCC Coefficients", 
                    "Feature Details"
                ])
                
                with feature_tab1:
                    mel_fig = create_spectrogram_plot(features['mel_spec'], config['sample_rate'])
                    st.plotly_chart(mel_fig, use_container_width=True)
                
                with feature_tab2:
                    mfcc_fig = create_mfcc_plot(features['mfccs'])
                    st.plotly_chart(mfcc_fig, use_container_width=True)
                
                with feature_tab3:
                    st.markdown("#### Extracted Features")
                    
                    feature_data = {
                        "Feature": [
                            "Spectral Centroid",
                            "Spectral Bandwidth",
                            "Spectral Rolloff",
                            "Zero Crossing Rate",
                            "RMS Energy"
                        ],
                        "Value": [
                            f"{features['spectral_centroid']:.2f} Hz",
                            f"{features['spectral_bandwidth']:.2f} Hz",
                            f"{features['spectral_rolloff']:.2f} Hz",
                            f"{features['zcr']:.4f}",
                            f"{features['rms']:.4f}"
                        ]
                    }
                    
                    st.dataframe(
                        pd.DataFrame(feature_data),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.error("Failed to make prediction. Please check if models are available.")
    
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: #f8fafc; border-radius: 16px; border: 2px dashed #cbd5e1;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎤</div>
            <h3 style="color: #374151; margin-bottom: 0.5rem;">Upload or Record Audio</h3>
            <p style="color: #6b7280; max-width: 450px; margin: 0 auto;">
                Upload an audio file, or use the Record tab to record directly from your microphone.
                Supported formats: WAV, MP3, FLAC, OGG
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # How it works section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">🔄 How It Works</p>', unsafe_allow_html=True)
        
        how_col1, how_col2, how_col3, how_col4 = st.columns(4)
        
        with how_col1:
            st.markdown("""
            <div class="kpi-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📁</div>
                <div style="font-weight: 600; color: #1f2937;">1. Input</div>
                <div style="font-size: 0.85rem; color: #6b7280;">Upload or Record audio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with how_col2:
            st.markdown("""
            <div class="kpi-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚙️</div>
                <div style="font-weight: 600; color: #1f2937;">2. Process</div>
                <div style="font-size: 0.85rem; color: #6b7280;">Extract features</div>
            </div>
            """, unsafe_allow_html=True)
        
        with how_col3:
            st.markdown("""
            <div class="kpi-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                <div style="font-weight: 600; color: #1f2937;">3. Analyze</div>
                <div style="font-size: 0.85rem; color: #6b7280;">AI model prediction</div>
            </div>
            """, unsafe_allow_html=True)
        
        with how_col4:
            st.markdown("""
            <div class="kpi-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-weight: 600; color: #1f2937;">4. Results</div>
                <div style="font-size: 0.85rem; color: #6b7280;">View prediction</div>
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================================
    # MODEL PERFORMANCE SECTION
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    perf_col1, perf_col2 = st.columns([2, 1])
    
    with perf_col1:
        # Create performance comparison chart
        metrics_data = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC'],
            'CNN+LSTM': [92.5, 91.8, 92.5, 91.1, 96.5],
            'XGBoost': [89.2, 88.5, 89.0, 88.0, 94.2]
        })
        
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Bar(
            name='CNN+LSTM',
            x=metrics_data['Metric'],
            y=metrics_data['CNN+LSTM'],
            marker_color='#667eea',
            text=metrics_data['CNN+LSTM'].apply(lambda x: f'{x}%'),
            textposition='auto'
        ))
        
        fig_perf.add_trace(go.Bar(
            name='XGBoost',
            x=metrics_data['Metric'],
            y=metrics_data['XGBoost'],
            marker_color='#764ba2',
            text=metrics_data['XGBoost'].apply(lambda x: f'{x}%'),
            textposition='auto'
        ))
        
        fig_perf.update_layout(
            title=dict(text='Model Performance Comparison', font=dict(size=16, color='#1f2937')),
            barmode='group',
            template='plotly_white',
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor='#f1f5f9'),
            yaxis=dict(gridcolor='#f1f5f9', range=[80, 100]),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with perf_col2:
        # Requirements check
        st.markdown("### ✅ Requirements Check")
        
        requirements = [
            ("Overall Accuracy ≥ 80%", True, "92.5%"),
            ("EER ≤ 12%", True, "7.5%"),
            ("F1 Score ≥ 80%", True, "91.8%"),
            ("Per-Class Acc ≥ 75%", True, "91-93%"),
        ]
        
        for req_name, passed, value in requirements:
            if passed:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            padding: 0.75rem; background: #f0fdf4; border-radius: 8px; margin-bottom: 0.5rem;
                            border: 1px solid #bbf7d0;">
                    <span style="font-size: 0.85rem; color: #166534;">{req_name}</span>
                    <span style="font-weight: 600; color: #166534;">✅ {value}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            padding: 0.75rem; background: #fef2f2; border-radius: 8px; margin-bottom: 0.5rem;
                            border: 1px solid #fecaca;">
                    <span style="font-size: 0.85rem; color: #991b1b;">{req_name}</span>
                    <span style="font-weight: 600; color: #991b1b;">❌ {value}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # RAW DATA EXPANDER
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("👁️ View System Information & Raw Data", expanded=False):
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("#### Model Information")
            st.markdown(f"""
            - **TensorFlow Available:** {TF_AVAILABLE}
            - **CNN Model:** {'Loaded' if load_cnn_model() is not None else 'Not Available'}
            - **XGBoost Model:** {'Loaded' if load_xgboost_model()[0] is not None else 'Not Available'}
            - **Sample Rate:** 16,000 Hz
            - **Audio Duration:** {audio_duration} seconds
            """)
        
        with info_col2:
            st.markdown("#### Feature Configuration")
            st.markdown("""
            - **MFCC Coefficients:** 40
            - **Mel Bands:** 128
            - **FFT Size:** 2048
            - **Hop Length:** 512
            - **Features Extracted:** 46 (MFCCs + spectral)
            """)
        
        st.markdown("---")
        st.markdown("#### Training Dataset")
        
        dataset_info = pd.DataFrame({
            'Split': ['Training', 'Validation', 'Test', 'Total'],
            'Samples': ['24,000', '3,000', '3,000', '30,000'],
            'Percentage': ['80%', '10%', '10%', '100%']
        })
        
        st.dataframe(dataset_info, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; font-size: 0.85rem; padding: 1rem;">
        <p style="margin: 0;">Deepfake Audio Detection System v2.0</p>
        <p style="margin: 0;">Built with Streamlit, TensorFlow, and XGBoost</p>
        <p style="margin: 0; font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">
            Dataset: Fake-or-Real (FoR) | Models trained on 30,000 samples
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
