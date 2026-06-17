"""
Test ResNet18 Training Modes
=============================

Tests both training modes for ResNet18MCDropout:
1. Feature-space mode (freeze_backbone=True) - Only train classifier
2. End-to-end mode (freeze_backbone=False) - Train entire network

This test validates Issue #1: Test ResNet Training Modes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest
from pathlib import Path

# Import the model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from uqlab.models.factory import ResNet18MCDropout


class TestResNetTrainingModes:
    """Test suite for ResNet18 training modes."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy CIFAR-10 style data for testing."""
        # 100 samples, 3 channels, 32x32 images
        X = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 10, (100,))
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        return dataloader
    
    def test_feature_space_mode_freezes_backbone(self):
        """Test that feature-space mode freezes backbone parameters."""
        model = ResNet18MCDropout(
            num_classes=10,
            dropout=0.3,
            pretrained=False,
            freeze_backbone=True
        )
        
        # Check that backbone parameters are frozen
        backbone_params_frozen = all(
            not param.requires_grad 
            for param in model.backbone.parameters()
        )
        assert backbone_params_frozen, "Backbone parameters should be frozen in feature-space mode"
        
        # Check that classifier parameters are trainable
        classifier_params_trainable = all(
            param.requires_grad 
            for param in model.classifier.parameters()
        )
        assert classifier_params_trainable, "Classifier parameters should be trainable"
        
        print("✅ Feature-space mode: Backbone frozen, classifier trainable")
    
    def test_end_to_end_mode_trains_all(self):
        """Test that end-to-end mode trains all parameters."""
        model = ResNet18MCDropout(
            num_classes=10,
            dropout=0.3,
            pretrained=False,
            freeze_backbone=False
        )
        
        # Check that all parameters are trainable
        all_params_trainable = all(
            param.requires_grad 
            for param in model.parameters()
        )
        assert all_params_trainable, "All parameters should be trainable in end-to-end mode"
        
        print("✅ End-to-end mode: All parameters trainable")
    
    def test_feature_space_training_step(self, dummy_data):
        """Test that feature-space mode can perform training step."""
        model = ResNet18MCDropout(
            num_classes=10,
            dropout=0.3,
            pretrained=False,
            freeze_backbone=True
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        # Get one batch
        X, y = next(iter(dummy_data))
        
        # Forward pass
        logits = model(X)
        loss = criterion(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that only classifier was updated
        # (backbone gradients should be None or zero)
        backbone_has_gradients = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.backbone.parameters()
        )
        assert not backbone_has_gradients, "Backbone should not have gradients in feature-space mode"
        
        classifier_has_gradients = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.classifier.parameters()
        )
        assert classifier_has_gradients, "Classifier should have gradients"
        
        print(f"✅ Feature-space training step: Loss={loss.item():.4f}")
    
    def test_end_to_end_training_step(self, dummy_data):
        """Test that end-to-end mode can perform training step."""
        model = ResNet18MCDropout(
            num_classes=10,
            dropout=0.3,
            pretrained=False,
            freeze_backbone=False
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        # Get one batch
        X, y = next(iter(dummy_data))
        
        # Forward pass
        logits = model(X)
        loss = criterion(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that both backbone and classifier were updated
        backbone_has_gradients = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.backbone.parameters()
        )
        assert backbone_has_gradients, "Backbone should have gradients in end-to-end mode"
        
        classifier_has_gradients = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.classifier.parameters()
        )
        assert classifier_has_gradients, "Classifier should have gradients"
        
        print(f"✅ End-to-end training step: Loss={loss.item():.4f}")
    
    def test_mc_dropout_inference(self, dummy_data):
        """Test MC Dropout inference works in both modes."""
        for freeze_backbone in [True, False]:
            mode_name = "feature-space" if freeze_backbone else "end-to-end"
            
            model = ResNet18MCDropout(
                num_classes=10,
                dropout=0.3,
                pretrained=False,
                freeze_backbone=freeze_backbone
            )
            
            model.eval()
            
            # Get one batch
            X, y = next(iter(dummy_data))
            
            # MC Dropout inference
            mc_probs = model.mc_forward(X, n_passes=5)
            
            # Check shape: [n_passes, batch_size, num_classes]
            assert mc_probs.shape == (5, X.shape[0], 10), \
                f"MC Dropout output shape incorrect for {mode_name} mode"
            
            # Check probabilities sum to 1
            prob_sums = mc_probs.sum(dim=-1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
                f"Probabilities should sum to 1 for {mode_name} mode"
            
            print(f"✅ MC Dropout inference works in {mode_name} mode")
    
    def test_feature_extraction(self, dummy_data):
        """Test feature extraction works in both modes."""
        for freeze_backbone in [True, False]:
            mode_name = "feature-space" if freeze_backbone else "end-to-end"
            
            model = ResNet18MCDropout(
                num_classes=10,
                dropout=0.3,
                pretrained=False,
                freeze_backbone=freeze_backbone
            )
            
            model.eval()
            
            # Get one batch
            X, y = next(iter(dummy_data))
            
            # Extract features
            with torch.no_grad():
                features = model.extract_features(X)
            
            # Check shape: [batch_size, 512] (ResNet18 feature dim)
            assert features.shape == (X.shape[0], 512), \
                f"Feature shape incorrect for {mode_name} mode"
            
            print(f"✅ Feature extraction works in {mode_name} mode: shape={features.shape}")
    
    def test_model_output_consistency(self, dummy_data):
        """Test that model outputs are consistent across modes."""
        # Create two models with same initialization
        torch.manual_seed(42)
        model_frozen = ResNet18MCDropout(
            num_classes=10,
            dropout=0.0,  # No dropout for deterministic test
            pretrained=False,
            freeze_backbone=True
        )
        
        torch.manual_seed(42)
        model_full = ResNet18MCDropout(
            num_classes=10,
            dropout=0.0,
            pretrained=False,
            freeze_backbone=False
        )
        
        # Get one batch
        X, y = next(iter(dummy_data))
        
        # Forward pass (eval mode, no dropout)
        model_frozen.eval()
        model_full.eval()
        
        with torch.no_grad():
            logits_frozen = model_frozen(X)
            logits_full = model_full(X)
        
        # Outputs should be identical (same initialization, no training yet)
        assert torch.allclose(logits_frozen, logits_full, atol=1e-5), \
            "Model outputs should be identical before training"
        
        print("✅ Model outputs consistent across modes before training")


def run_manual_test():
    """Run tests manually without pytest."""
    print("=" * 70)
    print("Testing ResNet18 Training Modes")
    print("=" * 70)
    print()
    
    # Create test instance
    test = TestResNetTrainingModes()
    
    # Create dummy data
    print("Creating dummy data...")
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"✅ Created {len(dataset)} samples\n")
    
    # Run tests
    tests = [
        ("Feature-space mode freezes backbone", test.test_feature_space_mode_freezes_backbone),
        ("End-to-end mode trains all parameters", test.test_end_to_end_mode_trains_all),
        ("Feature-space training step", lambda: test.test_feature_space_training_step(dataloader)),
        ("End-to-end training step", lambda: test.test_end_to_end_training_step(dataloader)),
        ("MC Dropout inference", lambda: test.test_mc_dropout_inference(dataloader)),
        ("Feature extraction", lambda: test.test_feature_extraction(dataloader)),
        ("Model output consistency", lambda: test.test_model_output_consistency(dataloader)),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}")
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ FAILED: {e}\n")
            failed += 1
    
    # Summary
    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 All tests passed! ResNet18 training modes work correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_manual_test()
    sys.exit(0 if success else 1)

# Made with Bob
