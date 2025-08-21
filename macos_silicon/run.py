#!/usr/bin/env python
"""
Main entry point for DanceBa on macOS Silicon
Provides a simple interface for dance generation from audio
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inference_macos import DancebaInferenceMacOS
import yaml
import numpy as np


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="DanceBa Dance Generation for macOS Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run.py music.wav
  
  # With custom model
  python run.py music.wav --model ./models/custom
  
  # Benchmark mode
  python run.py music.wav --benchmark
  
  # Force CPU usage
  python run.py music.wav --cpu
  
  # Custom configuration
  python run.py music.wav --config configs/custom.yaml
        """
    )
    
    # Required arguments
    parser.add_argument(
        'audio',
        type=str,
        help='Path to input audio file (WAV, MP3, etc.)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='./models',
        help='Path to model checkpoint directory'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/config_macos_silicon.yaml',
        help='Path to configuration file'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs',
        help='Output directory for generated dance'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['npz', 'json', 'both'],
        default='npz',
        help='Output format'
    )
    
    # Performance arguments
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage instead of MPS'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark'
    )
    
    parser.add_argument(
        '--benchmark-runs',
        type=int,
        default=10,
        help='Number of benchmark runs'
    )
    
    # Generation arguments
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (0.0-2.0)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling parameter'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=None,
        help='Maximum sequence length'
    )
    
    # Other arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run MPS tests before inference'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Check audio file
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return 1
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        config = {}
    
    # Run tests if requested
    if args.test:
        logger.info("Running MPS tests...")
        from tests.test_mps import main as test_main
        test_result = test_main()
        if test_result != 0:
            logger.warning("Some tests failed. Continuing anyway...")
    
    # Initialize model
    try:
        logger.info("Initializing model...")
        model = DancebaInferenceMacOS(
            model_path=args.model,
            use_mps=not args.cpu,
            memory_fraction=config.get('device', {}).get('memory_fraction', 0.7),
            mixed_precision=config.get('model', {}).get('use_mixed_precision', True)
        )
        logger.info(f"Model initialized on {model.device}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return 1
    
    # Run benchmark if requested
    if args.benchmark:
        logger.info(f"Running benchmark with {args.benchmark_runs} runs...")
        try:
            metrics = model.benchmark(args.audio, num_runs=args.benchmark_runs)
            
            print("\n" + "=" * 50)
            print("Benchmark Results")
            print("=" * 50)
            print(f"Device: {metrics['device']}")
            print(f"Average Time: {metrics['avg_time']:.3f} seconds")
            print(f"FPS: {metrics['fps']:.2f}")
            print("=" * 50)
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return 1
    
    # Run inference
    else:
        logger.info("Starting inference...")
        try:
            # Generate dance
            result = model.generate(
                audio_path=args.audio,
                temperature=args.temperature,
                top_k=args.top_k,
                max_length=args.max_length
            )
            
            # Create output directory
            os.makedirs(args.output, exist_ok=True)
            
            # Generate output filename
            audio_name = Path(args.audio).stem
            timestamp = Path(args.audio).stat().st_mtime
            from datetime import datetime
            time_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            output_base = f"{audio_name}_{time_str}"
            
            # Save outputs
            if args.format in ['npz', 'both']:
                output_path = os.path.join(args.output, f"{output_base}.npz")
                np.savez(output_path, **result)
                logger.info(f"Saved NPZ to {output_path}")
            
            if args.format in ['json', 'both']:
                import json
                # Convert numpy arrays to lists for JSON serialization
                json_result = {
                    'fps': float(result['fps']),
                    'num_frames': int(result['num_frames']),
                    'poses_shape': list(result['poses'].shape),
                    # Optionally include poses (may be large)
                    # 'poses': result['poses'].tolist()
                }
                
                output_path = os.path.join(args.output, f"{output_base}.json")
                with open(output_path, 'w') as f:
                    json.dump(json_result, f, indent=2)
                logger.info(f"Saved JSON metadata to {output_path}")
            
            # Print summary
            print("\n" + "=" * 50)
            print("Inference Complete")
            print("=" * 50)
            print(f"Audio: {args.audio}")
            print(f"Output: {args.output}/{output_base}")
            print(f"Frames: {result['num_frames']}")
            print(f"FPS: {result['fps']}")
            print(f"Duration: {result['num_frames'] / result['fps']:.2f} seconds")
            print("=" * 50)
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())