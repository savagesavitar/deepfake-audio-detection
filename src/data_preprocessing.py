"""
Data Preprocessing Module for Deepfake Audio Detection
Handles data loading, splitting, and augmentation.
"""
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import json


def create_dataset(
    data_dir: str,
    sr: int = 16000,
    duration: float = 3.0,
    max_files_per_class: Optional[int] = None
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Create dataset from directory structure.
    
    Expected structure:
    data_dir/
    ├── real/   (label 0)
    └── fake/   (label 1)
    
    Args:
        data_dir: Path to data directory
        sr: Sampling rate (for validation)
        duration: Expected duration in seconds
        max_files_per_class: Maximum files per class (None for all)
        
    Returns:
        Tuple of (file_paths, labels, class_distribution)
    """
    file_paths = []
    labels = []
    
    # Define class mapping
    class_map = {'real': 0, 'fake': 1}
    
    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        # Get all audio files
        files = [
            f for f in os.listdir(class_dir)
            if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))
        ]
        
        # Limit files if specified
        if max_files_per_class:
            files = files[:max_files_per_class]
        
        print(f"Found {len(files)} {class_name} files")
        
        for file_name in tqdm(files, desc=f"Processing {class_name}"):
            file_path = os.path.join(class_dir, file_name)
            file_paths.append(file_path)
            labels.append(label)
    
    # Calculate distribution
    unique, counts = np.unique(labels, return_counts=True)
    distribution = {class_names[i]: int(count) 
                   for i, class_names in enumerate(['real', 'fake'])
                   for i, count in zip(unique, counts)}
    
    return file_paths, labels, distribution


def load_audio_file(
    file_path: str,
    sr: int = 16000,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load a single audio file.
    
    Args:
        file_path: Path to audio file
        sr: Target sampling rate
        duration: Duration to load (None for full)
        
    Returns:
        Tuple of (audio array, sampling rate)
    """
    try:
        audio, orig_sr = librosa.load(file_path, sr=sr, duration=duration)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None


def preprocess_audio(
    audio: np.ndarray,
    sr: int = 16000,
    target_duration: float = 3.0,
    normalize: bool = True,
    remove_silence: bool = False
) -> np.ndarray:
    """
    Preprocess audio signal.
    
    Args:
        audio: Input audio array
        sr: Sampling rate
        target_duration: Target duration in seconds
        normalize: Normalize amplitude
        remove_silence: Remove silence
        
    Returns:
        Preprocessed audio array
    """
    # Remove silence
    if remove_silence:
        audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Normalize
    if normalize:
        audio = librosa.util.normalize(audio)
    
    # Target length
    target_length = int(sr * target_duration)
    
    # Pad or truncate
    if len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)))
    elif len(audio) > target_length:
        # Truncate
        audio = audio[:target_length]
    
    return audio


def split_dataset(
    file_paths: List[str],
    labels: List[int],
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        file_paths: List of file paths
        labels: List of labels
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        stratify: Maintain class distribution
        
    Returns:
        Dictionary with train, val, test splits
    """
    # First split: train+val and test
    stratify_labels = labels if stratify else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        file_paths, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )
    
    # Second split: train and val
    stratify_labels = y_train_val if stratify else None
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_labels
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def create_data_generator(
    file_paths: List[str],
    labels: List[int],
    batch_size: int = 32,
    sr: int = 16000,
    duration: float = 3.0,
    augment: bool = False,
    shuffle: bool = True
):
    """
    Create a data generator for training.
    
    Args:
        file_paths: List of file paths
        labels: List of labels
        batch_size: Batch size
        sr: Sampling rate
        duration: Audio duration
        augment: Apply data augmentation
        shuffle: Shuffle data
        
    Yields:
        Batches of (features, labels)
    """
    from feature_extraction import extract_mel_spectrogram, load_audio
    
    n_samples = len(file_paths)
    indices = np.arange(n_samples)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_features = []
            batch_labels = []
            
            for idx in batch_indices:
                try:
                    # Load and preprocess audio
                    audio, _ = load_audio(file_paths[idx], sr=sr, duration=duration)
                    audio = preprocess_audio(audio, sr, duration)
                    
                    # Extract mel spectrogram
                    mel_spec = extract_mel_spectrogram(audio, sr)
                    
                    # Resize to fixed shape
                    mel_spec = resize_spectrogram(mel_spec, (128, 128))
                    
                    # Add channel dimension
                    mel_spec = np.expand_dims(mel_spec, axis=-1)
                    
                    # Augmentation
                    if augment:
                        mel_spec = apply_augmentation(mel_spec)
                    
                    batch_features.append(mel_spec)
                    batch_labels.append(labels[idx])
                    
                except Exception as e:
                    print(f"Error processing {file_paths[idx]}: {str(e)}")
                    continue
            
            if batch_features:
                yield np.array(batch_features), np.array(batch_labels)


def resize_spectrogram(
    spec: np.ndarray,
    target_shape: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Resize spectrogram to target shape.
    
    Args:
        spec: Input spectrogram
        target_shape: Target (height, width)
        
    Returns:
        Resized spectrogram
    """
    from scipy.ndimage import zoom
    
    zoom_factors = (
        target_shape[0] / spec.shape[0],
        target_shape[1] / spec.shape[1]
    )
    
    return zoom(spec, zoom_factors, order=1)


def apply_augmentation(features: np.ndarray) -> np.ndarray:
    """
    Apply data augmentation to features.
    
    Args:
        features: Input features (height, width, channels)
        
    Returns:
        Augmented features
    """
    # Random noise injection
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 0.01, features.shape)
        features = features + noise
    
    # Random time shift
    if np.random.random() > 0.5:
        shift = np.random.randint(-10, 10)
        features = np.roll(features, shift, axis=1)
    
    # Random scaling
    if np.random.random() > 0.5:
        scale = np.random.uniform(0.9, 1.1)
        features = features * scale
    
    return features


def prepare_ml_features(
    file_paths: List[str],
    labels: List[int],
    sr: int = 16000,
    duration: float = 3.0,
    n_mfcc: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for classical ML models.
    
    Args:
        file_paths: List of file paths
        labels: List of labels
        sr: Sampling rate
        duration: Audio duration
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Tuple of (features, labels)
    """
    from feature_extraction import extract_mfcc, extract_spectral_features, load_audio
    
    features_list = []
    valid_labels = []
    
    for file_path, label in tqdm(zip(file_paths, labels), 
                                  total=len(file_paths),
                                  desc="Extracting ML features"):
        try:
            # Load audio
            audio, _ = load_audio(file_path, sr=sr, duration=duration)
            audio = preprocess_audio(audio, sr, duration)
            
            # Extract MFCC
            mfcc = extract_mfcc(audio, sr, n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Extract spectral features
            spectral = extract_spectral_features(audio, sr)
            spectral_values = np.array(list(spectral.values()))
            
            # Combine features
            combined = np.concatenate([mfcc_mean, mfcc_std, spectral_values])
            features_list.append(combined)
            valid_labels.append(label)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    return np.array(features_list), np.array(valid_labels)


def save_dataset_info(
    splits: Dict[str, Tuple[List[str], List[int]]],
    save_path: str
) -> None:
    """
    Save dataset information to JSON file.
    
    Args:
        splits: Dataset splits
        save_path: Path to save JSON
    """
    info = {}
    for split_name, (paths, labels) in splits.items():
        unique, counts = np.unique(labels, return_counts=True)
        info[split_name] = {
            'total': len(paths),
            'distribution': {str(k): int(v) for k, v in zip(unique, counts)}
        }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Dataset info saved to: {save_path}")
