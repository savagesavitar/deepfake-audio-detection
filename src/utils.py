"""
Deepfake Audio Detection - Utility Functions
Helper functions for audio processing, visualization, and evaluation.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from pathlib import Path
import soundfile as sf


def calculate_eer(y_true, y_score):
    """
    Calculate Equal Error Rate (EER).
    
    EER is the point where False Acceptance Rate (FAR) equals
    False Rejection Rate (FRR).
    
    Args:
        y_true: True labels (0 or 1)
        y_score: Predicted probabilities
        
    Returns:
        eer: Equal Error Rate
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def plot_confusion_matrix(y_true, y_pred, classes=['Fake', 'Real'], 
                          title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        title: Plot title
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return cm


def plot_roc_curve(y_true, y_prob, model_name='Model', save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fpr, tpr, roc_auc


def plot_training_history(history, save_path=None):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Keras training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_audio(audio, sr, title='Audio Waveform', save_path=None):
    """
    Visualize audio waveform.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_spectrogram(audio, sr, n_mels=128, title='Mel-Spectrogram', 
                          save_path=None):
    """
    Visualize mel-spectrogram.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of mel bands
        title: Plot title
        save_path: Path to save the plot
    """
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.title(title, fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_mfcc(audio, sr, n_mfcc=40, title='MFCC Coefficients', 
                   save_path=None):
    """
    Visualize MFCC coefficients.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        title: Plot title
        save_path: Path to save the plot
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.title(title, fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficient')
    plt.colorbar()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compare_features(real_features, fake_features, feature_names, 
                     title='Feature Comparison', save_path=None):
    """
    Compare features between real and fake audio.
    
    Args:
        real_features: Features from real audio
        fake_features: Features from fake audio
        feature_names: Names of features
        title: Plot title
        save_path: Path to save the plot
    """
    n_features = len(feature_names)
    fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, (name, real_vals, fake_vals) in enumerate(
            zip(feature_names, real_features, fake_features)):
        axes[i].hist(real_vals, bins=30, alpha=0.5, label='Real', density=True)
        axes[i].hist(fake_vals, bins=30, alpha=0.5, label='Fake', density=True)
        axes[i].set_title(name)
        axes[i].legend()
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def get_audio_info(audio_path):
    """
    Get audio file information.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        info: Dictionary with audio information
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None, duration=None)
        
        info = {
            'path': str(audio_path),
            'sample_rate': sr,
            'duration': len(audio) / sr,
            'samples': len(audio),
            'channels': 1 if audio.ndim == 1 else audio.shape[1],
            'file_size': Path(audio_path).stat().st_size / 1024  # KB
        }
        
        return info
    except Exception as e:
        print(f"Error getting audio info: {e}")
        return None


def batch_predict(audio_paths, model, scaler=None, config=None):
    """
    Batch prediction for multiple audio files.
    
    Args:
        audio_paths: List of audio file paths
        model: Trained model
        scaler: Feature scaler (optional)
        config: Model configuration
        
    Returns:
        results: List of prediction results
    """
    results = []
    
    for audio_path in audio_paths:
        try:
            # Load and preprocess
            audio, _ = librosa.load(audio_path, sr=config['sample_rate'],
                                   duration=config['duration'])
            audio = librosa.util.normalize(audio)
            audio, _ = librosa.effects.trim(audio, top_db=30)
            
            n_samples = config['sample_rate'] * config['duration']
            if len(audio) > n_samples:
                audio = audio[:n_samples]
            else:
                audio = np.pad(audio, (0, n_samples - len(audio)), mode='constant')
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=audio, sr=config['sample_rate'],
                                         n_mfcc=config['n_mfcc'])
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=config['sample_rate'],
                                                      n_mels=config['n_mels'])
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Prepare input
            X = mel_spec_db[..., np.newaxis]
            X = np.expand_dims(X, axis=0)
            
            # Predict
            prob = model.predict(X, verbose=0)[0][0]
            prediction = 'real' if prob > 0.5 else 'fake'
            confidence = prob if prob > 0.5 else 1 - prob
            
            results.append({
                'path': str(audio_path),
                'prediction': prediction,
                'confidence': float(confidence),
                'probability': float(prob)
            })
            
        except Exception as e:
            results.append({
                'path': str(audio_path),
                'error': str(e)
            })
    
    return results


def create_test_audio(output_dir, duration=3, sr=16000):
    """
    Create sample test audio files for testing.
    
    Args:
        output_dir: Directory to save test audio
        duration: Duration in seconds
        sr: Sample rate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sine wave (simulated "real" audio)
    t = np.linspace(0, duration, int(sr * duration), False)
    real_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    real_audio += 0.3 * np.sin(2 * np.pi * 880 * t)  # Add harmonic
    
    # Add some noise
    real_audio += 0.05 * np.random.randn(len(real_audio))
    
    # Save
    sf.write(str(output_dir / 'test_real.wav'), real_audio, sr)
    
    # Create different pattern (simulated "fake" audio)
    fake_audio = 0.5 * np.sin(2 * np.pi * 330 * t)  # Different frequency
    fake_audio += 0.4 * np.sin(2 * np.pi * 660 * t)
    fake_audio += 0.1 * np.random.randn(len(fake_audio))
    
    # Save
    sf.write(str(output_dir / 'test_fake.wav'), fake_audio, sr)
    
    print(f"Test audio files created in {output_dir}")
    print(f"  - test_real.wav")
    print(f"  - test_fake.wav")
    
    return output_dir


# Example usage
if __name__ == "__main__":
    # Create test audio
    create_test_audio('test_samples')
    
    # Example: Calculate EER
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.3, 0.8, 0.7, 0.2, 0.9, 0.4, 0.6, 0.85, 0.15])
    eer = calculate_eer(y_true, y_score)
    print(f"\nExample EER: {eer:.4f}")
    
    # Example: Get audio info
    info = get_audio_info('test_samples/test_real.wav')
    if info:
        print(f"\nAudio Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
