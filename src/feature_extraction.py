"""
Feature Extraction Module for Deepfake Audio Detection
Extracts MFCCs, Mel-spectrograms, and spectral features from audio files.
"""
import numpy as np
import librosa
from typing import Dict, Tuple, Optional


def load_audio(
    file_path: str,
    sr: int = 16000,
    duration: Optional[float] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file.
    
    Args:
        file_path: Path to audio file
        sr: Target sampling rate
        duration: Duration to load (None for full audio)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio array, sampling rate)
    """
    try:
        audio, orig_sr = librosa.load(
            file_path,
            sr=sr,
            duration=duration,
            mono=mono
        )
        # Normalize amplitude
        audio = librosa.util.normalize(audio)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Error loading audio file {file_path}: {str(e)}")


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract Mel-Frequency Cepstral Coefficients.
    
    Args:
        audio: Audio signal array
        sr: Sampling rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length
        
    Returns:
        MFCC features (n_mfcc, time_steps)
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # Add delta and delta-delta features
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Concatenate all MFCC features
    mfcc_combined = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
    return mfcc_combined


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: int = 0,
    fmax: Optional[int] = None
) -> np.ndarray:
    """
    Extract Mel spectrogram.
    
    Args:
        audio: Audio signal array
        sr: Sampling rate
        n_mels: Number of Mel bands
        n_fft: FFT window size
        hop_length: Hop length
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Log-scaled Mel spectrogram (n_mels, time_steps)
    """
    if fmax is None:
        fmax = sr // 2
        
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def extract_spectral_features(
    audio: np.ndarray,
    sr: int = 16000
) -> Dict[str, float]:
    """
    Extract spectral features from audio.
    
    Args:
        audio: Audio signal array
        sr: Sampling rate
        
    Returns:
        Dictionary of spectral features
    """
    features = {}
    
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr
    )[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr
    )[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr
    )[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr
    )
    features['spectral_contrast_mean'] = np.mean(spectral_contrast)
    features['spectral_contrast_std'] = np.std(spectral_contrast)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    return features


def extract_chroma_features(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract chroma features (pitch class profiles).
    
    Args:
        audio: Audio signal array
        sr: Sampling rate
        n_fft: FFT window size
        hop_length: Hop length
        
    Returns:
        Chroma features (12, time_steps)
    """
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return chroma


def extract_all_features(
    file_path: str,
    sr: int = 16000,
    duration: float = 3.0,
    target_shape: Tuple[int, int] = (128, 128)
) -> Dict[str, np.ndarray]:
    """
    Extract all features from an audio file.
    
    Args:
        file_path: Path to audio file
        sr: Sampling rate
        duration: Duration to load in seconds
        target_shape: Target shape for mel spectrogram
        
    Returns:
        Dictionary containing all extracted features
    """
    # Load audio
    audio, sr = load_audio(file_path, sr=sr, duration=duration)
    
    # Pad or truncate to fixed length
    target_length = sr * duration
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    # Extract features
    features = {}
    
    # MFCCs (flatten for ML models)
    mfcc = extract_mfcc(audio, sr)
    features['mfcc'] = mfcc
    features['mfcc_flat'] = np.mean(mfcc, axis=1)
    
    # Mel spectrogram (for CNN)
    mel_spec = extract_mel_spectrogram(audio, sr)
    # Resize to target shape
    mel_spec = resize_spectrogram(mel_spec, target_shape)
    features['mel_spectrogram'] = mel_spec
    
    # Spectral features
    features['spectral'] = extract_spectral_features(audio, sr)
    
    # Chroma features
    features['chroma'] = extract_chroma_features(audio, sr)
    
    return features


def resize_spectrogram(
    spectrogram: np.ndarray,
    target_shape: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Resize spectrogram to target shape using interpolation.
    
    Args:
        spectrogram: Input spectrogram
        target_shape: Target (height, width)
        
    Returns:
        Resized spectrogram
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors
    zoom_factors = (
        target_shape[0] / spectrogram.shape[0],
        target_shape[1] / spectrogram.shape[1]
    )
    
    # Resize
    resized = zoom(spectrogram, zoom_factors, order=1)
    return resized


def prepare_features_for_model(
    features: Dict[str, np.ndarray],
    model_type: str = 'cnn'
) -> np.ndarray:
    """
    Prepare features for model input.
    
    Args:
        features: Dictionary of extracted features
        model_type: 'cnn' for mel spectrogram, 'ml' for flat features
        
    Returns:
        Prepared feature array
    """
    if model_type == 'cnn':
        # For CNN: add channel dimension
        mel_spec = features['mel_spectrogram']
        return np.expand_dims(mel_spec, axis=-1)
    else:
        # For ML: concatenate all flat features
        mfcc_flat = features['mfcc_flat']
        spectral = np.array(list(features['spectral'].values()))
        return np.concatenate([mfcc_flat, spectral])
