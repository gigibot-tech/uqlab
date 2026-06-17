"""
Basic Usage Example for UQ Benchmarks Package

This script demonstrates how to:
1. Load CIFAR-10 with epistemic and aleatoric manipulation
2. Train a Gaussian Logits model
3. Evaluate uncertainties
4. Run a benchmark sweep
"""

import numpy as np
from uqlab.evaluation.benchmarks.data.cifar10 import (
    get_cifar10_dataset,
    get_cifar10_with_epistemic_manipulation
)
from uqlab.evaluation.benchmarks.models.gaussian_logits import create_gaussian_logits_method


def example_1_basic_training():
    """Example 1: Basic training on CIFAR-10"""
    print("=" * 60)
    print("Example 1: Basic Training on CIFAR-10")
    print("=" * 60)
    
    # Load dataset (test mode for quick demo)
    print("\n1. Loading CIFAR-10 dataset (test mode)...")
    dataset = get_cifar10_dataset(test_mode=True, noise_rate=0.2)
    print(f"   Training samples: {len(dataset.X_train)}")
    print(f"   Test samples: {len(dataset.X_test)}")
    print(f"   Noise rate: {dataset.noise_rate:.1%}")
    
    # Create method
    print("\n2. Creating Gaussian Logits method...")
    method = create_gaussian_logits_method(num_samples=10)
    print(f"   Method: {method.name}")
    print(f"   Framework: {method.framework}")
    
    # Train
    print("\n3. Training model...")
    train_config = {
        'epochs': 2,  # Quick demo
        'batch_size': 32,
        'verbose': 1
    }
    model = method.train(dataset, train_config)
    print("   Training complete!")
    
    # Evaluate
    print("\n4. Evaluating uncertainties...")
    eval_config = {
        'batch_size': 32,
        'num_samples': 10
    }
    accuracy, aleatoric, epistemic = method.evaluate(model, dataset, eval_config)
    
    print(f"\n   Results:")
    print(f"   - Accuracy: {accuracy:.3f}")
    print(f"   - Aleatoric Uncertainty: {aleatoric:.3f}")
    print(f"   - Epistemic Uncertainty: {epistemic:.3f}")


def example_2_epistemic_manipulation():
    """Example 2: Epistemic uncertainty manipulation"""
    print("\n" + "=" * 60)
    print("Example 2: Epistemic Uncertainty Manipulation")
    print("=" * 60)
    
    # Create dataset with under-supported classes
    print("\n1. Creating dataset with epistemic manipulation...")
    print("   Under-supporting classes 3 (cat) and 5 (dog)")
    
    dataset = get_cifar10_with_epistemic_manipulation(
        under_supported_classes=[3, 5],
        under_train_per_class=30,      # Very few samples
        regular_train_per_class=100,   # Normal samples
        eval_per_class=50,             # Balanced test
        noise_rate=0.0,                # No aleatoric noise
        seed=42
    )
    
    # Check class distribution
    print("\n   Training class distribution:")
    for class_idx in range(10):
        count = (dataset.y_train == class_idx).sum()
        status = "UNDER-SUPPORTED" if class_idx in [3, 5] else "regular"
        print(f"   Class {class_idx}: {count:3d} samples ({status})")
    
    print(f"\n   Total training: {len(dataset.y_train)} samples")
    print(f"   Total test: {len(dataset.y_test)} samples (balanced)")
    
    # Train and evaluate
    print("\n2. Training Gaussian Logits model...")
    method = create_gaussian_logits_method()
    
    accuracy, aleatoric, epistemic = method.train_and_evaluate(
        dataset,
        train_config={'epochs': 2, 'batch_size': 32, 'verbose': 0},
        eval_config={'batch_size': 32}
    )
    
    print(f"\n   Results:")
    print(f"   - Accuracy: {accuracy:.3f}")
    print(f"   - Aleatoric: {aleatoric:.3f} (should be LOW - no label noise)")
    print(f"   - Epistemic: {epistemic:.3f} (should be HIGH - class imbalance)")


def example_3_label_noise_benchmark():
    """Example 3: Label noise benchmark sweep"""
    print("\n" + "=" * 60)
    print("Example 3: Label Noise Benchmark Sweep")
    print("=" * 60)
    
    print("\n1. Setting up benchmark...")
    print("   Sweeping noise rates: [0.0, 0.2, 0.4]")
    
    # Create method
    method = create_gaussian_logits_method()
    
    # Define dataset generator
    def dataset_gen(noise_rate):
        return get_cifar10_dataset(
            test_mode=True,  # Quick demo
            noise_rate=noise_rate,
            seed=42
        )
    
    # Run benchmark
    print("\n2. Running benchmark...")
    results = method.run_benchmark(
        dataset_generator=dataset_gen,
        parameter_values=[0.0, 0.2, 0.4],
        train_config={'epochs': 2, 'batch_size': 32, 'verbose': 0},
        eval_config={'batch_size': 32}
    )
    
    # Display results
    print("\n   Results:")
    print("   " + "-" * 50)
    print(f"   {'Noise Rate':<12} {'Accuracy':<10} {'Aleatoric':<12} {'Epistemic':<12}")
    print("   " + "-" * 50)
    
    for i, noise_rate in enumerate(results.changed_parameter_values):
        acc = results.accuracies[i]
        ale = results.aleatoric_uncertainties[i]
        epi = results.epistemic_uncertainties[i]
        print(f"   {noise_rate:<12.1f} {acc:<10.3f} {ale:<12.3f} {epi:<12.3f}")
    
    print("\n   Expected behavior:")
    print("   - Accuracy should DECREASE with more noise")
    print("   - Aleatoric should INCREASE with more noise (C1 criterion)")
    print("   - Epistemic should stay relatively CONSTANT")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("UQ Benchmarks Package - Usage Examples")
    print("=" * 60)
    
    try:
        # Check if Keras is available
        import keras
        print("\n✅ Keras is available - running examples...")
        
        # Run examples
        example_1_basic_training()
        example_2_epistemic_manipulation()
        example_3_label_noise_benchmark()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except ImportError:
        print("\n❌ Keras not available!")
        print("Install with: pip install keras tensorflow")
        print("\nShowing what the examples would do...")
        
        # Show structure without running
        print("\nExample 1: Basic Training")
        print("  - Load CIFAR-10 with 20% noise")
        print("  - Train Gaussian Logits model")
        print("  - Evaluate uncertainties")
        
        print("\nExample 2: Epistemic Manipulation")
        print("  - Create class imbalance (cats/dogs under-supported)")
        print("  - Train and evaluate")
        print("  - Expect high epistemic uncertainty")
        
        print("\nExample 3: Label Noise Benchmark")
        print("  - Sweep noise rates: 0%, 20%, 40%")
        print("  - Validate C1 criterion (aleatoric increases with noise)")
        print("  - Generate benchmark results")


if __name__ == "__main__":
    main()

# Made with Bob
