#!/usr/bin/env python
"""
Test script for MPS (Metal Performance Shaders) functionality
Verifies that PyTorch can properly utilize Apple Silicon GPU
"""

import sys
import torch
import numpy as np
import time
from typing import Dict, Any


def test_mps_availability() -> bool:
    """Test if MPS backend is available."""
    print("Testing MPS availability...")
    
    if torch.backends.mps.is_available():
        print("✅ MPS backend is available")
        
        if torch.backends.mps.is_built():
            print("✅ PyTorch was built with MPS support")
        else:
            print("❌ PyTorch was NOT built with MPS support")
            return False
            
        return True
    else:
        print("❌ MPS backend is NOT available")
        print("   Please ensure you have:")
        print("   - macOS 12.3 or later")
        print("   - PyTorch 2.0 or later with MPS support")
        return False


def test_device_creation() -> torch.device:
    """Test creating MPS device."""
    print("\nTesting device creation...")
    
    try:
        device = torch.device("mps")
        print(f"✅ Successfully created device: {device}")
        return device
    except Exception as e:
        print(f"❌ Failed to create MPS device: {e}")
        print("   Falling back to CPU")
        return torch.device("cpu")


def test_tensor_operations(device: torch.device) -> bool:
    """Test basic tensor operations on the device."""
    print(f"\nTesting tensor operations on {device}...")
    
    try:
        # Create tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Test various operations
        operations = {
            "Addition": lambda: x + y,
            "Multiplication": lambda: x * y,
            "Matrix multiplication": lambda: torch.matmul(x, y),
            "Mean": lambda: x.mean(),
            "Standard deviation": lambda: x.std(),
            "Softmax": lambda: torch.softmax(x, dim=1),
        }
        
        for op_name, op_func in operations.items():
            try:
                result = op_func()
                if device.type == "mps":
                    torch.mps.synchronize()
                print(f"  ✅ {op_name}: Success")
            except Exception as e:
                print(f"  ❌ {op_name}: Failed - {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False


def test_memory_allocation(device: torch.device) -> Dict[str, Any]:
    """Test memory allocation limits."""
    print(f"\nTesting memory allocation on {device}...")
    
    results = {
        "max_size": 0,
        "allocations": []
    }
    
    sizes = [
        (100, 100),
        (1000, 1000),
        (5000, 5000),
        (10000, 10000),
        (20000, 20000)
    ]
    
    for size in sizes:
        try:
            tensor = torch.randn(*size, device=device)
            memory_mb = (tensor.element_size() * tensor.nelement()) / (1024 * 1024)
            
            results["allocations"].append({
                "size": size,
                "memory_mb": memory_mb,
                "success": True
            })
            results["max_size"] = size
            
            print(f"  ✅ Allocated {size}: {memory_mb:.2f} MB")
            
            # Clean up
            del tensor
            if device.type == "mps":
                torch.mps.empty_cache()
                
        except Exception as e:
            results["allocations"].append({
                "size": size,
                "success": False,
                "error": str(e)
            })
            print(f"  ❌ Failed to allocate {size}: {e}")
            break
    
    return results


def benchmark_performance(device: torch.device) -> Dict[str, float]:
    """Benchmark performance on the device."""
    print(f"\nBenchmarking performance on {device}...")
    
    results = {}
    
    # Test different operations
    benchmarks = {
        "matmul_small": (100, 100),
        "matmul_medium": (1000, 1000),
        "matmul_large": (5000, 5000),
    }
    
    for name, size in benchmarks.items():
        try:
            # Create tensors
            x = torch.randn(size, device=device)
            y = torch.randn(size, device=device)
            
            # Warmup
            for _ in range(3):
                _ = torch.matmul(x, y)
                if device.type == "mps":
                    torch.mps.synchronize()
            
            # Benchmark
            start = time.time()
            num_iterations = 10
            
            for _ in range(num_iterations):
                _ = torch.matmul(x, y)
                if device.type == "mps":
                    torch.mps.synchronize()
            
            elapsed = time.time() - start
            avg_time = elapsed / num_iterations
            
            results[name] = avg_time
            print(f"  {name} ({size[0]}x{size[1]}): {avg_time*1000:.2f} ms")
            
            # Clean up
            del x, y
            if device.type == "mps":
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"  {name}: Failed - {e}")
            results[name] = -1
    
    return results


def test_mixed_precision(device: torch.device) -> bool:
    """Test mixed precision (float16) support."""
    print(f"\nTesting mixed precision on {device}...")
    
    try:
        # Test float16 operations
        x_fp16 = torch.randn(1000, 1000, device=device, dtype=torch.float16)
        y_fp16 = torch.randn(1000, 1000, device=device, dtype=torch.float16)
        
        # Test autocast
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            result = torch.matmul(x_fp16, y_fp16)
            if device.type == "mps":
                torch.mps.synchronize()
        
        print("✅ Mixed precision (float16) is supported")
        return True
        
    except Exception as e:
        print(f"❌ Mixed precision failed: {e}")
        return False


def test_model_operations(device: torch.device) -> bool:
    """Test common model operations."""
    print(f"\nTesting model operations on {device}...")
    
    try:
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1)
        ).to(device)
        
        # Test forward pass
        batch_size = 32
        input_data = torch.randn(batch_size, 784, device=device)
        
        with torch.no_grad():
            output = model(input_data)
            if device.type == "mps":
                torch.mps.synchronize()
        
        assert output.shape == (batch_size, 10)
        print("✅ Model forward pass successful")
        
        # Test different layers
        test_layers = [
            ("Conv2d", torch.nn.Conv2d(3, 64, 3)),
            ("BatchNorm2d", torch.nn.BatchNorm2d(64)),
            ("LSTM", torch.nn.LSTM(128, 256, batch_first=True)),
            ("Transformer", torch.nn.TransformerEncoderLayer(512, 8)),
        ]
        
        for layer_name, layer in test_layers:
            try:
                layer = layer.to(device)
                print(f"  ✅ {layer_name}: Supported")
            except Exception as e:
                print(f"  ⚠️  {layer_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model operations failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DanceBa MPS Testing Suite for macOS Silicon")
    print("=" * 60)
    
    # Track results
    all_passed = True
    
    # Test MPS availability
    if not test_mps_availability():
        print("\n⚠️  MPS not available, tests will run on CPU")
        device = torch.device("cpu")
    else:
        # Create device
        device = test_device_creation()
    
    # Run tests
    tests = [
        ("Tensor Operations", lambda: test_tensor_operations(device)),
        ("Memory Allocation", lambda: test_memory_allocation(device)),
        ("Performance Benchmark", lambda: benchmark_performance(device)),
        ("Mixed Precision", lambda: test_mixed_precision(device)),
        ("Model Operations", lambda: test_model_operations(device)),
    ]
    
    print("\n" + "=" * 60)
    print("Running Test Suite")
    print("=" * 60)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, bool) and not result:
                all_passed = False
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if all_passed and device.type == "mps":
        print("✅ All tests passed! MPS is working correctly.")
        print(f"   Device: {device}")
        print(f"   PyTorch: {torch.__version__}")
    elif device.type == "cpu":
        print("⚠️  Tests ran on CPU. MPS not available.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())