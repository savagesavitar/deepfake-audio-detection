"""
Model Architecture Module for Deepfake Audio Detection
Defines CNN+LSTM hybrid model and classical ML models.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from typing import Tuple, Optional


def build_cnn_lstm_model(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    num_classes: int = 1,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Build CNN + LSTM hybrid model for deepfake audio detection.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes (1 for binary)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Block 1: CNN Feature Extraction
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Reshape for LSTM
        layers.Reshape((-1, 256)),
        
        # LSTM for temporal patterns
        layers.LSTM(128, return_sequences=True, dropout=0.3),
        layers.LSTM(64, return_sequences=False, dropout=0.3),
        
        # Dense classifier
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    num_classes: int = 1
) -> keras.Model:
    """
    Build pure CNN model (lighter alternative).
    
    Args:
        input_shape: Input shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_transfer_learning_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 1,
    base_model_name: str = 'MobileNetV2'
) -> keras.Model:
    """
    Build transfer learning model using pre-trained network.
    
    Args:
        input_shape: Input shape (must be 3 channels)
        num_classes: Number of output classes
        base_model_name: Name of base model
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Convert grayscale to RGB if needed
    if input_shape[-1] == 1:
        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
    else:
        x = inputs
    
    # Base model
    if base_model_name == 'MobileNetV2':
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'ResNet50':
        base_model = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    base_model.trainable = False  # Freeze base model
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(
    patience: int = 10,
    model_save_path: str = 'models/best_model.keras'
) -> list:
    """
    Get training callbacks.
    
    Args:
        patience: Early stopping patience
        model_save_path: Path to save best model
        
    Returns:
        List of callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    return callbacks


def augment_data(
    x: np.ndarray,
    y: np.ndarray,
    augmentation_factor: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to training data.
    
    Args:
        x: Input features (batch, height, width, channels)
        y: Labels
        augmentation_factor: Number of augmented versions
        
    Returns:
        Augmented data and labels
    """
    x_augmented = [x]
    y_augmented = [y]
    
    for _ in range(augmentation_factor - 1):
        # Add random noise
        noise = np.random.normal(0, 0.01, x.shape)
        x_augmented.append(x + noise)
        y_augmented.append(y)
        
        # Random time shift (shift spectrogram)
        shifted = np.roll(x, shift=np.random.randint(-10, 10), axis=2)
        x_augmented.append(shifted)
        y_augmented.append(y)
    
    return np.concatenate(x_augmented), np.concatenate(y_augmented)
