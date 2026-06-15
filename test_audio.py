"""
Deepfake Audio Detection - Test Script
Test new audio samples with trained models.

Usage:
    python test_audio.py --audio_path path/to/audio.wav
    python test_audio.py --audio_path path/to/audio.wav --model cnn
    python test_audio.py --audio_path path/to/audio.wav --model xgboost
"""

import argparse
import numpy as np
import librosa
import joblib
from pathlib import Path

# Try to import TensorFlow with custom path
try:
    import sys
    # Ensure system numpy loads first
    try:
        import numpy as _np_check
        del _np_check
    except ImportError:
        pass
    
    # Try standard import first
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ImportError:
        tf_path = r"C:\tf_install"
        if tf_path not in sys.path:
            sys.path.insert(0, tf_path)
        import tensorflow as tf
        TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN model won't work.")


class DeepfakeDetector:
    """Deepfake Audio Detection System."""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.config = self._load_config()
        self.scaler = None
        self.cnn_model = None
        self.xgb_model = None
        
    def _load_config(self):
        """Load model configuration."""
        config_path = self.models_dir / 'config.pkl'
        if config_path.exists():
            return joblib.load(config_path)
        else:
            return {
                'sample_rate': 16000,
                'duration': 3,
                'n_mfcc': 40,
                'n_mels': 128,
                'hop_length': 512,
                'n_fft': 2048
            }
    
    def load_models(self, model_type="cnn"):
        """Load specified model."""
        if model_type == "cnn" and TF_AVAILABLE:
            # Try .keras first, then .h5
            model_path = self.models_dir / 'deepfake_cnn_lstm_final.keras'
            if not model_path.exists():
                model_path = self.models_dir / 'deepfake_cnn_lstm_final.h5'
            if model_path.exists():
                self.cnn_model = tf.keras.models.load_model(model_path)
                print(f"CNN model loaded from {model_path}")
            else:
                print(f"CNN model not found at {model_path}")
                return False
        elif model_type == "xgboost":
            model_path = self.models_dir / 'deepfake_xgboost.pkl'
            if model_path.exists():
                self.xgb_model = joblib.load(model_path)
                print(f"XGBoost model loaded from {model_path}")
            else:
                print(f"XGBoost model not found at {model_path}")
                return False
            
            scaler_path = self.models_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
        
        return True
    
    def preprocess_audio(self, audio_path):
        """Load and preprocess audio file."""
        try:
            # Load audio
            audio, _ = librosa.load(
                audio_path,
                sr=self.config['sample_rate'],
                duration=self.config['duration']
            )
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=30)
            
            # Pad or truncate
            n_samples = self.config['sample_rate'] * self.config['duration']
            if len(audio) > n_samples:
                audio = audio[:n_samples]
            else:
                audio = np.pad(audio, (0, n_samples - len(audio)), mode='constant')
            
            return audio
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
    
    def extract_features(self, audio):
        """Extract features from audio."""
        sr = self.config['sample_rate']
        
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.config['n_mfcc'])
        features['mfccs_mean'] = np.mean(mfccs, axis=1)
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.config['n_mels'])
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spec'] = mel_spec_db
        
        # Spectral features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        features['rms'] = np.mean(librosa.feature.rms(y=audio))
        
        return features
    
    def predict(self, audio_path, model_type="cnn"):
        """Predict if audio is real or fake."""
        # Preprocess audio
        audio = self.preprocess_audio(audio_path)
        if audio is None:
            return None
        
        # Extract features
        features = self.extract_features(audio)
        
        # Make prediction based on model type
        if model_type == "cnn" and self.cnn_model is not None:
            # Prepare input for CNN
            mel_spec = features['mel_spec']
            mel_spec = mel_spec[..., np.newaxis]
            mel_spec = np.expand_dims(mel_spec, axis=0)
            
            # Predict
            prob = self.cnn_model.predict(mel_spec, verbose=0)[0][0]
            
        elif model_type == "xgboost" and self.xgb_model is not None:
            # Prepare input for XGBoost
            other_features = [
                features['spectral_centroid'],
                features['spectral_bandwidth'],
                features['spectral_rolloff'],
                features['spectral_contrast'],
                features['zcr'],
                features['rms']
            ]
            X = np.hstack([features['mfccs_mean'], other_features]).reshape(1, -1)
            
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Predict
            prob = self.xgb_model.predict_proba(X)[0][1]
        
        else:
            print(f"Model {model_type} not loaded. Please load a model first.")
            return None
        
        # Determine prediction
        prediction = 'real' if prob > 0.5 else 'fake'
        confidence = prob if prob > 0.5 else 1 - prob
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'probability_real': float(prob),
            'probability_fake': float(1 - prob)
        }


def main():
    parser = argparse.ArgumentParser(description='Deepfake Audio Detection')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'xgboost'],
                        help='Model type to use (default: cnn)')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing trained models')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not Path(args.audio_path).exists():
        print(f"Error: Audio file not found: {args.audio_path}")
        return
    
    # Initialize detector
    detector = DeepfakeDetector(models_dir=args.models_dir)
    
    # Load model
    print(f"Loading {args.model} model...")
    if not detector.load_models(args.model):
        print("Failed to load model. Please ensure models are trained.")
        return
    
    # Make prediction
    print(f"\nAnalyzing: {args.audio_path}")
    print("-" * 50)
    
    result = detector.predict(args.audio_path, model_type=args.model)
    
    if result:
        print(f"\nPrediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"\nProbability Real: {result['probability_real']*100:.2f}%")
        print(f"Probability Fake: {result['probability_fake']*100:.2f}%")
        
        # Interpretation
        print("\n" + "="*50)
        if result['prediction'] == 'real':
            print("INTERPRETATION: This audio appears to be GENUINE (Human)")
        else:
            print("INTERPRETATION: This audio appears to be DEEPFAKE (AI-Generated)")
        
        if result['confidence'] >= 0.9:
            print("CONFIDENCE LEVEL: Very High")
        elif result['confidence'] >= 0.75:
            print("CONFIDENCE LEVEL: High")
        elif result['confidence'] >= 0.6:
            print("CONFIDENCE LEVEL: Medium")
        else:
            print("CONFIDENCE LEVEL: Low - Results may be unreliable")
    else:
        print("Prediction failed. Please check the audio file.")


if __name__ == "__main__":
    main()
