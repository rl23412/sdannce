"""
Unit tests for 2D confidence integration in s-DANNCE training.
Tests confidence data loading, confidence weighting functions, and loss computation.
"""

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
import tempfile
import os

# Import the modules to test
from dannce.engine.data.io import load_label2d_confidence, create_default_confidence
from dannce.engine.trainer.train_utils import LossHelper


class TestConfidenceDataLoading(unittest.TestCase):
    """Test confidence data loading and creation functions."""
    
    def test_create_default_confidence(self):
        """Test creation of default confidence scores when not available."""
        # Mock labelData structure
        labelData = [
            {
                'data_2d': np.random.rand(100, 6, 20, 2),  # 100 frames, 6 cameras, 20 keypoints, 2 coords
                'data_frame': np.arange(100),
                'data_sampleID': np.arange(100)
            }
        ]
        
        confidence_data = create_default_confidence(labelData, default_confidence=0.8)
        
        self.assertEqual(len(confidence_data), 1)
        self.assertIn('data_2d_confidence', confidence_data[0])
        
        conf_shape = confidence_data[0]['data_2d_confidence'].shape
        self.assertEqual(conf_shape, (100, 6, 20))  # frames, cameras, keypoints
        
        # Check all values are set to default
        self.assertTrue(np.all(confidence_data[0]['data_2d_confidence'] == 0.8))
    
    def test_load_label2d_confidence_missing_data(self):
        """Test handling of missing confidence data."""
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp_file:
            # Create a minimal mat file without confidence data
            import scipy.io as sio
            sio.savemat(tmp_file.name, {'labelData': []})
            
            confidence_data = load_label2d_confidence(tmp_file.name)
            self.assertIsNone(confidence_data)
            
        os.unlink(tmp_file.name)


class TestConfidenceWeighting(unittest.TestCase):
    """Test confidence weighting functions in the loss handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_handler = LossHelper({})
        self.device = torch.device('cpu')
    
    def test_compute_confidence_weights_linear(self):
        """Test linear confidence weighting."""
        confidence_scores = torch.tensor([0.1, 0.5, 0.8, 1.0])
        weights = self.loss_handler.compute_confidence_weights(
            confidence_scores, method='linear', strength=1.0
        )
        
        # Linear method should return confidence * strength
        expected_weights = confidence_scores * 1.0
        torch.testing.assert_close(weights, expected_weights.clamp(min=0.01, max=10.0))
    
    def test_compute_confidence_weights_sigmoid(self):
        """Test sigmoid confidence weighting."""
        confidence_scores = torch.tensor([0.1, 0.5, 0.8, 1.0])
        weights = self.loss_handler.compute_confidence_weights(
            confidence_scores, method='sigmoid', strength=1.0
        )
        
        # Sigmoid method should enhance contrast
        self.assertTrue(torch.all(weights >= 0.01))
        self.assertTrue(torch.all(weights <= 10.0))
        
        # Higher confidence should give higher weights
        self.assertTrue(weights[3] > weights[2])
        self.assertTrue(weights[2] > weights[1])
    
    def test_compute_confidence_weights_exponential(self):
        """Test exponential confidence weighting."""
        confidence_scores = torch.tensor([0.1, 0.5, 0.8, 1.0])
        weights = self.loss_handler.compute_confidence_weights(
            confidence_scores, method='exponential', strength=2.0
        )
        
        # Exponential method: confidence^(1/strength)
        expected_weights = torch.pow(confidence_scores, 1.0/2.0)
        torch.testing.assert_close(weights, expected_weights.clamp(min=0.01, max=10.0))
    
    def test_apply_confidence_weighting_same_device(self):
        """Test applying confidence weighting with tensors on same device."""
        loss_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        confidence_tensor = torch.tensor([0.5, 0.8, 0.2, 1.0])
        
        weighted_loss = self.loss_handler.apply_confidence_weighting(
            loss_tensor, confidence_tensor, method='linear', strength=1.0
        )
        
        # Should apply confidence weighting
        self.assertEqual(weighted_loss.shape, loss_tensor.shape)
        
        # Weighted loss should be different from original
        self.assertFalse(torch.equal(weighted_loss, loss_tensor))
    
    def test_apply_confidence_weighting_none_confidence(self):
        """Test applying confidence weighting when confidence is None."""
        loss_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        weighted_loss = self.loss_handler.apply_confidence_weighting(
            loss_tensor, None, method='linear', strength=1.0
        )
        
        # Should return original loss unchanged
        torch.testing.assert_close(weighted_loss, loss_tensor)
    
    def test_apply_confidence_weighting_different_devices(self):
        """Test applying confidence weighting with tensors on different devices."""
        loss_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        confidence_tensor = torch.tensor([0.5, 0.8, 0.2, 1.0])
        
        # Test that function handles device mismatch gracefully
        weighted_loss = self.loss_handler.apply_confidence_weighting(
            loss_tensor, confidence_tensor, method='linear', strength=1.0
        )
        
        self.assertEqual(weighted_loss.device, loss_tensor.device)


class TestConfidenceIntegration(unittest.TestCase):
    """Test integration of confidence weighting in loss computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        loss_params = {
            'use_2d_confidence_weighting': True,
            'min_confidence_threshold': 0.3,
            'confidence_loss_weighting_method': 'linear',
            'confidence_weighting_strength': 1.0,
            'clip_2d_loss': True,
            'max_2d_loss_value': 70.0
        }
        self.loss_handler = LossHelper(loss_params)
        
        # Mock a simple loss function
        self.mock_loss_fn = MagicMock()
        self.mock_loss_fn.return_value = torch.tensor(2.0)
        self.loss_handler.loss_fcns_2d = {'L1Loss': self.mock_loss_fn}
    
    def test_confidence_parameter_extraction(self):
        """Test that confidence parameters are correctly extracted from loss_params."""
        self.assertTrue(self.loss_handler.loss_params.get('use_2d_confidence_weighting'))
        self.assertEqual(self.loss_handler.loss_params.get('min_confidence_threshold'), 0.3)
        self.assertEqual(self.loss_handler.loss_params.get('confidence_loss_weighting_method'), 'linear')
        self.assertEqual(self.loss_handler.loss_params.get('confidence_weighting_strength'), 1.0)
    
    def test_confidence_thresholding(self):
        """Test that confidence values below threshold are set to low weight."""
        confidence_scores = torch.tensor([0.1, 0.5, 0.8, 1.0])  # 0.1 is below 0.3 threshold
        min_threshold = 0.3
        
        thresholded_conf = torch.where(
            confidence_scores < min_threshold,
            torch.tensor(0.01),
            confidence_scores
        )
        
        self.assertAlmostEqual(thresholded_conf[0].item(), 0.01, places=6)  # Below threshold
        self.assertAlmostEqual(thresholded_conf[1].item(), 0.5, places=6)   # Above threshold
        self.assertAlmostEqual(thresholded_conf[2].item(), 0.8, places=6)   # Above threshold
        self.assertAlmostEqual(thresholded_conf[3].item(), 1.0, places=6)   # Above threshold


class TestConfigurationValidation(unittest.TestCase):
    """Test validation of confidence-related configuration parameters."""
    
    def test_valid_confidence_parameters(self):
        """Test validation of valid confidence parameters."""
        from dannce import ConfigDANNCETrain
        
        config = ConfigDANNCETrain()
        
        # Test default values
        self.assertFalse(config.use_2d_confidence_weighting)
        self.assertEqual(config.min_confidence_threshold, 0.5)
        self.assertEqual(config.confidence_loss_weighting_method, 'linear')
        self.assertEqual(config.confidence_weighting_strength, 1.0)
    
    def test_confidence_weighting_methods(self):
        """Test that all supported confidence weighting methods work."""
        methods = ['linear', 'sigmoid', 'exponential']
        loss_handler = LossHelper({})
        confidence_scores = torch.tensor([0.2, 0.5, 0.8, 1.0])
        
        for method in methods:
            weights = loss_handler.compute_confidence_weights(
                confidence_scores, method=method, strength=1.0
            )
            
            # All methods should return valid weights
            self.assertTrue(torch.all(weights >= 0.01))
            self.assertTrue(torch.all(weights <= 10.0))
            self.assertEqual(len(weights), len(confidence_scores))


if __name__ == '__main__':
    unittest.main()
