"""
Training Module for Deepfake Audio Detection
Handles model training, validation, and saving.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from typing import Tuple, Dict, Optional, List
from tqdm import tqdm

from src.feature_extraction import (
    extract_mel_spectrogram,
    extract_mfcc,
    load_audio,
    resize_spectrogram
)
from src.data_preprocessing import preprocess_audio
from src.model import build_cnn_lstm_model, get_callbacks
from src.evaluate import evaluate_model, calculate_eer


def prepare_cnn_data(
    file_paths: List[str],
    labels: List[int],
    sr: int = 16000,
    duration: float = 3.0,
    target_shape: Tuple[int, int] = (128, 128),
    augment: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for CNN training.
    
    Args:
        file_paths: List of audio file paths
        labels: List of labels
        sr: Sampling rate
        duration: Audio duration
        target_shape: Target spectrogram shape
        augment: Apply data augmentation
        
    Returns:
        Tuple of (features, labels)
    """
    features = []
    valid_labels = []
    
    for file_path, label in tqdm(zip(file_paths, labels),
                                  total=len(file_paths),
                                  desc="Preparing CNN data"):
        try:
            # Load and preprocess audio
            audio, _ = load_audio(file_path, sr=sr, duration=duration)
            audio = preprocess_audio(audio, sr, duration)
            
            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(audio, sr)
            mel_spec = resize_spectrogram(mel_spec, target_shape)
            
            # Add channel dimension
            mel_spec = np.expand_dims(mel_spec, axis=-1)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            features.append(mel_spec)
            valid_labels.append(label)
            
            # Data augmentation
            if augment:
                # Add noisy version
                noise = np.random.normal(0, 0.01, mel_spec.shape)
                features.append(mel_spec + noise)
                valid_labels.append(label)
                
                # Add shifted version
                shifted = np.roll(mel_spec, shift=np.random.randint(-5, 5), axis=1)
                features.append(shifted)
                valid_labels.append(label)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    return np.array(features), np.array(valid_labels)


def prepare_ml_data(
    file_paths: List[str],
    labels: List[int],
    sr: int = 16000,
    duration: float = 3.0,
    n_mfcc: int = 40
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare data for classical ML training.
    
    Args:
        file_paths: List of audio file paths
        labels: List of labels
        sr: Sampling rate
        duration: Audio duration
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Tuple of (features, labels, scaler)
    """
    from src.feature_extraction import extract_spectral_features
    
    features_list = []
    valid_labels = []
    
    for file_path, label in tqdm(zip(file_paths, labels),
                                  total=len(file_paths),
                                  desc="Preparing ML data"):
        try:
            # Load and preprocess audio
            audio, _ = load_audio(file_path, sr=sr, duration=duration)
            audio = preprocess_audio(audio, sr, duration)
            
            # Extract MFCCs
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
    
    X = np.array(features_list)
    y = np.array(valid_labels)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler


def train_cnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_save_path: str = 'models/deepfake_cnn_lstm.h5',
    epochs: int = 50,
    batch_size: int = 32,
    use_class_weights: bool = True
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """
    Train CNN+LSTM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_save_path: Path to save model
        epochs: Number of epochs
        batch_size: Batch size
        use_class_weights: Use class weights for imbalance
        
    Returns:
        Tuple of (trained model, training history)
    """
    print("\n" + "="*60)
    print("TRAINING CNN+LSTM MODEL")
    print("="*60)
    
    # Build model
    input_shape = X_train.shape[1:]
    model = build_cnn_lstm_model(input_shape=input_shape)
    model.summary()
    
    # Calculate class weights
    class_weights = None
    if use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"\nClass weights: {class_weights}")
    
    # Get callbacks
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    callbacks = get_callbacks(patience=10, model_save_path=model_save_path)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    return model, history.history


def train_ml_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'xgboost',
    model_save_path: str = 'models/deepfake_ml_model.pkl'
) -> Tuple[object, Dict[str, float]]:
    """
    Train classical ML model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: Type of model ('xgboost', 'svm', 'rf')
        model_save_path: Path to save model
        
    Returns:
        Tuple of (trained model, validation metrics)
    """
    print("\n" + "="*60)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("="*60)
    
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    elif model_type == 'svm':
        from sklearn.svm import SVC
        model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
    elif model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_scores = model.predict_proba(X_val)[:, 1]
    
    metrics = evaluate_model(y_val, y_pred, y_scores)
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_macro']:.4f}")
    if 'eer' in metrics:
        print(f"  EER: {metrics['eer']:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    return model, metrics


def evaluate_trained_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    class_names: List[str] = ['Genuine', 'Deepfake']
) -> Dict[str, float]:
    """
    Evaluate trained model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        scaler: Feature scaler (for ML models)
        class_names: Class names
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    # Get predictions
    if isinstance(model, keras.Model):
        y_scores = model.predict(X_test, verbose=0).flatten()
    else:
        y_scores = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_scores >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = evaluate_model(y_test, y_pred, y_scores, class_names)
    
    # Print results
    print(f"\nTest Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1 Score: {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)")
    if 'eer' in metrics:
        print(f"  EER: {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {metrics['per_class_accuracy'][i]*100:.2f}%")
    
    return metrics


def full_training_pipeline(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: List[str],
    test_labels: List[int],
    models_dir: str = 'models',
    use_cnn: bool = True,
    use_ml: bool = True,
    ml_model_type: str = 'xgboost'
) -> Dict[str, object]:
    """
    Complete training pipeline.
    
    Args:
        train_paths: Training file paths
        train_labels: Training labels
        val_paths: Validation file paths
        val_labels: Validation labels
        test_paths: Test file paths
        test_labels: Test labels
        models_dir: Directory to save models
        use_cnn: Train CNN model
        use_ml: Train ML model
        ml_model_type: ML model type
        
    Returns:
        Dictionary with trained models and metrics
    """
    results = {}
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Train CNN model
    if use_cnn:
        print("\nPreparing CNN data...")
        X_train_cnn, y_train_cnn = prepare_cnn_data(
            train_paths, train_labels, augment=True
        )
        X_val_cnn, y_val_cnn = prepare_cnn_data(val_paths, val_labels)
        X_test_cnn, y_test_cnn = prepare_cnn_data(test_paths, test_labels)
        
        # Save prepared data
        np.save(os.path.join(models_dir, 'X_train_cnn.npy'), X_train_cnn)
        np.save(os.path.join(models_dir, 'X_val_cnn.npy'), X_val_cnn)
        np.save(os.path.join(models_dir, 'X_test_cnn.npy'), X_test_cnn)
        
        cnn_model, history = train_cnn_model(
            X_train_cnn, y_train_cnn,
            X_val_cnn, y_val_cnn,
            model_save_path=os.path.join(models_dir, 'deepfake_cnn_lstm.h5')
        )
        
        cnn_metrics = evaluate_trained_model(
            cnn_model, X_test_cnn, y_test_cnn
        )
        
        results['cnn_model'] = cnn_model
        results['cnn_history'] = history
        results['cnn_metrics'] = cnn_metrics
    
    # Train ML model
    if use_ml:
        print("\nPreparing ML data...")
        X_train_ml, y_train_ml, scaler = prepare_ml_data(
            train_paths, train_labels
        )
        X_val_ml, y_val_ml, _ = prepare_ml_data(val_paths, val_labels)
        X_test_ml, y_test_ml, _ = prepare_ml_data(test_paths, test_labels)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
        
        # Save prepared data
        np.save(os.path.join(models_dir, 'X_train_ml.npy'), X_train_ml)
        np.save(os.path.join(models_dir, 'X_val_ml.npy'), X_val_ml)
        np.save(os.path.join(models_dir, 'X_test_ml.npy'), X_test_ml)
        
        ml_model, ml_metrics = train_ml_model(
            X_train_ml, y_train_ml,
            X_val_ml, y_val_ml,
            model_type=ml_model_type,
            model_save_path=os.path.join(
                models_dir, f'deepfake_{ml_model_type}.pkl'
            )
        )
        
        ml_test_metrics = evaluate_trained_model(
            ml_model, X_test_ml, y_test_ml, scaler
        )
        
        results['ml_model'] = ml_model
        results['ml_metrics'] = ml_test_metrics
    
    return results
