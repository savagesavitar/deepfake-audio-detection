"""
Deepfake Audio Detection - Generate Sample Test Audio
Creates sample audio files for testing the detection system.

Usage:
    python create_test_samples.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def create_test_audio(output_dir="test_samples", duration=3, sr=16000):
    """
    Create sample test audio files for testing.
    
    Creates synthetic audio samples that simulate real and fake audio patterns.
    Note: These are simplified examples for testing. Real deepfake detection
    requires analysis of actual human speech and AI-generated audio.
    
    Args:
        output_dir: Directory to save test audio
        duration: Duration in seconds
        sr: Sample rate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # =====================================================
    # Create "Real" Audio Sample (simulated human speech)
    # =====================================================
    # Simulate speech-like patterns with multiple harmonics
    real_audio = np.zeros_like(t)
    
    # Fundamental frequency (varies like human speech)
    f0 = 150 + 50 * np.sin(2 * np.pi * 2 * t)  # Pitch variation
    real_audio += 0.4 * np.sin(2 * np.pi * f0 * t)
    
    # Harmonics
    real_audio += 0.2 * np.sin(2 * np.pi * 2 * f0 * t)
    real_audio += 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    
    # Add envelope (simulates speech rhythm)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    real_audio *= envelope
    
    # Add natural noise (low level)
    real_audio += 0.02 * np.random.randn(len(t))
    
    # Normalize
    real_audio = real_audio / np.max(np.abs(real_audio)) * 0.8
    
    # Save
    sf.write(str(output_dir / 'sample_real.wav'), real_audio, sr)
    print(f"Created: {output_dir / 'sample_real.wav'}")
    
    # =====================================================
    # Create "Fake" Audio Sample (simulated AI-generated)
    # =====================================================
    # Simulate more robotic/synthetic patterns
    fake_audio = np.zeros_like(t)
    
    # More uniform frequency (less natural variation)
    f0 = 200  # Fixed pitch
    fake_audio += 0.5 * np.sin(2 * np.pi * f0 * t)
    
    # Strong harmonics (typical of some TTS)
    fake_audio += 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
    fake_audio += 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    fake_audio += 0.1 * np.sin(2 * np.pi * 4 * f0 * t)
    
    # Less natural envelope
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 1.5 * t)
    fake_audio *= envelope
    
    # Add synthetic artifacts (slight modulation)
    fake_audio *= (1 + 0.1 * np.sin(2 * np.pi * 50 * t))
    
    # Add less natural noise
    fake_audio += 0.01 * np.random.randn(len(t))
    
    # Normalize
    fake_audio = fake_audio / np.max(np.abs(fake_audio)) * 0.8
    
    # Save
    sf.write(str(output_dir / 'sample_fake.wav'), fake_audio, sr)
    print(f"Created: {output_dir / 'sample_fake.wav'}")
    
    # =====================================================
    # Create Additional Test Samples
    # =====================================================
    
    # Noisy real audio
    noisy_real = real_audio + 0.1 * np.random.randn(len(t))
    noisy_real = noisy_real / np.max(np.abs(noisy_real)) * 0.8
    sf.write(str(output_dir / 'sample_real_noisy.wav'), noisy_real, sr)
    print(f"Created: {output_dir / 'sample_real_noisy.wav'}")
    
    # Clean fake audio
    clean_fake = fake_audio.copy()
    clean_fake = clean_fake / np.max(np.abs(clean_fake)) * 0.8
    sf.write(str(output_dir / 'sample_fake_clean.wav'), clean_fake, sr)
    print(f"Created: {output_dir / 'sample_fake_clean.wav'}")
    
    print("\n" + "="*60)
    print("Test audio samples created successfully!")
    print("="*60)
    print(f"\nFiles created in: {output_dir.resolve()}")
    print("\nYou can test these with:")
    print("  python test_audio.py --audio_path test_samples/sample_real.wav")
    print("  python test_audio.py --audio_path test_samples/sample_fake.wav")
    
    return output_dir


if __name__ == "__main__":
    create_test_audio()
