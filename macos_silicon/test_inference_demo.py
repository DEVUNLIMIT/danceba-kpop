#!/usr/bin/env python
"""
Demo inference test for macOS Silicon
Tests the infrastructure without requiring the full model
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import os
import sys
from pathlib import Path

def test_audio_loading(audio_path):
    """Test loading and processing audio file."""
    print(f"\n{'='*60}")
    print("Testing Audio Loading and Processing")
    print(f"{'='*60}")
    
    # Check file
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    print(f"✅ Audio file found: {audio_path}")
    file_size = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"   File size: {file_size:.2f} MB")
    
    # Load audio
    try:
        # Load with librosa
        print("\nLoading audio with librosa...")
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        print(f"✅ Audio loaded successfully")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Shape: {audio.shape}")
        print(f"   Min/Max values: {audio.min():.4f} / {audio.max():.4f}")
        
        return audio, sr
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return None

def test_feature_extraction(audio, sr):
    """Test audio feature extraction."""
    print(f"\n{'='*60}")
    print("Testing Feature Extraction")
    print(f"{'='*60}")
    
    try:
        # Extract various features
        features = {}
        
        # 1. Mel spectrogram
        print("Extracting mel spectrogram...")
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        features['mel_spec'] = mel_spec
        print(f"✅ Mel spectrogram: shape {mel_spec.shape}")
        
        # 2. MFCC
        print("Extracting MFCC...")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc'] = mfcc
        print(f"✅ MFCC: shape {mfcc.shape}")
        
        # 3. Tempo and beat
        print("Extracting tempo...")
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        features['beats'] = beats
        print(f"✅ Tempo: {features['tempo']:.1f} BPM")
        print(f"✅ Beat frames: {len(beats)}")
        
        # 4. Chroma features
        print("Extracting chroma features...")
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma'] = chroma
        print(f"✅ Chroma: shape {chroma.shape}")
        
        return features
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None

def test_mps_processing(features):
    """Test processing features on MPS device."""
    print(f"\n{'='*60}")
    print("Testing MPS Processing")
    print(f"{'='*60}")
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using MPS device")
    else:
        device = torch.device("cpu")
        print(f"⚠️  MPS not available, using CPU")
    
    try:
        # Convert features to tensors
        print("\nConverting features to tensors...")
        tensors = {}
        
        for name, feat in features.items():
            if isinstance(feat, np.ndarray):
                tensor = torch.from_numpy(feat).float().to(device)
                tensors[name] = tensor
                print(f"  ✅ {name}: {tensor.shape} on {device}")
        
        # Simulate some processing
        print("\nSimulating dance generation pipeline...")
        
        # 1. Encoder simulation
        mel_tensor = tensors['mel_spec']
        batch_mel = mel_tensor.unsqueeze(0)  # Add batch dimension
        
        # Simple conv simulation
        if device.type == "mps":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                conv = torch.nn.Conv2d(1, 64, 3, padding=1).to(device)
                encoded = conv(batch_mel.unsqueeze(1))
                print(f"✅ Encoder output: {encoded.shape}")
        else:
            conv = torch.nn.Conv2d(1, 64, 3, padding=1).to(device)
            encoded = conv(batch_mel.unsqueeze(1))
            print(f"✅ Encoder output: {encoded.shape}")
        
        # 2. Generation simulation
        seq_length = 100
        pose_dim = 263  # Standard pose dimension
        
        print(f"\nGenerating mock dance sequence...")
        dance_sequence = torch.randn(1, seq_length, pose_dim, device=device)
        print(f"✅ Generated sequence: {dance_sequence.shape}")
        print(f"   Frames: {seq_length}")
        print(f"   Pose dimensions: {pose_dim}")
        print(f"   FPS: 30")
        print(f"   Duration: {seq_length/30:.2f} seconds")
        
        return dance_sequence
        
    except Exception as e:
        print(f"❌ MPS processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results(dance_sequence, output_path):
    """Save the results."""
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}")
    
    try:
        # Convert to numpy
        result_np = dance_sequence.cpu().numpy()
        
        # Save as npz
        np.savez(
            output_path,
            poses=result_np,
            fps=30.0,
            num_frames=result_np.shape[1]
        )
        
        print(f"✅ Results saved to: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return False

def main():
    """Main demo function."""
    print(f"\n{'='*60}")
    print("DanceBa macOS Silicon Inference Demo")
    print(f"{'='*60}")
    
    # Get audio file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "../test_data/atheart_goodgirl.mp3"
    
    print(f"\nInput: {audio_path}")
    
    # 1. Load audio
    result = test_audio_loading(audio_path)
    if result is None:
        return 1
    audio, sr = result
    
    # 2. Extract features
    features = test_feature_extraction(audio, sr)
    if features is None:
        return 1
    
    # 3. Process with MPS
    dance_sequence = test_mps_processing(features)
    if dance_sequence is None:
        return 1
    
    # 4. Save results
    output_path = f"outputs/demo_{Path(audio_path).stem}.npz"
    os.makedirs("outputs", exist_ok=True)
    
    if not save_results(dance_sequence, output_path):
        return 1
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ Demo Complete!")
    print(f"{'='*60}")
    print(f"Input: {audio_path}")
    print(f"Output: {output_path}")
    print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())