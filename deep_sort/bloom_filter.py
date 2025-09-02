# vim: expandtab:ts=4:sw=4
"""
Bloom Filter implementation for Deep SORT pedestrian tracking.

This module provides a Bloom filter optimized for storing feature vectors
and detection hashes to improve tracking performance and reduce memory usage.
"""

import numpy as np
import hashlib
from typing import List, Tuple, Optional
import math


class BloomFilter:
    """
    A Bloom filter implementation optimized for Deep SORT tracking.
    
    This Bloom filter can store feature vectors, detection hashes, and other
    tracking-related data with configurable false positive rates.
    
    Parameters
    ----------
    expected_elements : int
        Expected number of elements to be stored in the filter.
    false_positive_rate : float
        Desired false positive rate (0.0 to 1.0).
    feature_dim : int, optional
        Dimension of feature vectors if storing features directly.
    hash_functions : int, optional
        Number of hash functions to use. If None, calculated automatically.
    bit_array_size : int, optional
        Size of the bit array. If None, calculated automatically.
    """
    
    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01,
                 feature_dim: Optional[int] = None, hash_functions: Optional[int] = None,
                 bit_array_size: Optional[int] = None):
        
        if not 0 < false_positive_rate < 1:
            raise ValueError("False positive rate must be between 0 and 1")
        
        if expected_elements <= 0:
            raise ValueError("Expected elements must be positive")
        
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate
        self.feature_dim = feature_dim
        
        # Calculate optimal parameters if not provided
        if bit_array_size is None:
            self.bit_array_size = self._calculate_optimal_size(expected_elements, false_positive_rate)
        else:
            self.bit_array_size = bit_array_size
            
        if hash_functions is None:
            self.hash_functions = self._calculate_optimal_hash_functions(expected_elements, self.bit_array_size)
        else:
            self.hash_functions = hash_functions
        
        # Initialize bit array
        self.bit_array = np.zeros(self.bit_array_size, dtype=np.uint8)
        self.element_count = 0
        
        # Pre-compute hash seeds for consistent hashing
        # Use deterministic seeds for consistent results across runs
        np.random.seed(42)  # Fixed seed for reproducibility
        self.hash_seeds = np.random.randint(0, 2**32, size=self.hash_functions, dtype=np.uint32)
        
    def _calculate_optimal_size(self, n: int, p: float) -> int:
        """Calculate optimal bit array size for given parameters."""
        return int(-n * math.log(p) / (math.log(2) ** 2))
    
    def _calculate_optimal_hash_functions(self, n: int, m: int) -> int:
        """Calculate optimal number of hash functions for given parameters."""
        return int(m / n * math.log(2))
    
    def _get_hash_values(self, item: bytes) -> List[int]:
        """Generate hash values for an item using multiple hash functions."""
        hash_values = []
        
        for seed in self.hash_seeds:
            # Use seeded hash for consistent results
            hash_obj = hashlib.sha256()
            hash_obj.update(seed.tobytes())
            hash_obj.update(item)
            hash_value = int.from_bytes(hash_obj.digest(), byteorder='big')
            hash_values.append(hash_value % self.bit_array_size)
            
        return hash_values
    
    def _feature_to_bytes(self, feature: np.ndarray) -> bytes:
        """Convert feature vector to bytes for hashing."""
        if self.feature_dim is not None and feature.shape[0] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.feature_dim}, got {feature.shape[0]}")
        
        # Normalize and quantize feature for consistent hashing
        normalized = feature / (np.linalg.norm(feature) + 1e-8)
        quantized = np.round(normalized * 1000).astype(np.int32)
        return quantized.tobytes()
    
    def _detection_to_bytes(self, detection_data: dict) -> bytes:
        """Convert detection data to bytes for hashing."""
        # Create a hashable representation of detection data
        detection_str = f"{detection_data.get('bbox', '')}_{detection_data.get('confidence', '')}_{detection_data.get('frame_idx', '')}"
        return detection_str.encode('utf-8')
    
    def add_feature(self, feature: np.ndarray) -> None:
        """
        Add a feature vector to the Bloom filter.
        
        Parameters
        ----------
        feature : np.ndarray
            Feature vector to add.
        """
        feature_bytes = self._feature_to_bytes(feature)
        hash_values = self._get_hash_values(feature_bytes)
        
        for hash_val in hash_values:
            self.bit_array[hash_val] = 1
            
        self.element_count += 1
    
    def add_detection(self, detection_data: dict) -> None:
        """
        Add detection data to the Bloom filter.
        
        Parameters
        ----------
        detection_data : dict
            Dictionary containing detection information (bbox, confidence, frame_idx, etc.)
        """
        detection_bytes = self._detection_to_bytes(detection_data)
        hash_values = self._get_hash_values(detection_bytes)
        
        for hash_val in hash_values:
            self.bit_array[hash_val] = 1
            
        self.element_count += 1
    
    def add_raw_bytes(self, data: bytes) -> None:
        """
        Add raw bytes data to the Bloom filter.
        
        Parameters
        ----------
        data : bytes
            Raw bytes data to add.
        """
        hash_values = self._get_hash_values(data)
        
        for hash_val in hash_values:
            self.bit_array[hash_val] = 1
            
        self.element_count += 1
    
    def contains_feature(self, feature: np.ndarray) -> bool:
        """
        Check if a feature vector is likely in the Bloom filter.
        
        Parameters
        ----------
        feature : np.ndarray
            Feature vector to check.
            
        Returns
        -------
        bool
            True if the feature is likely in the filter (may have false positives).
        """
        feature_bytes = self._feature_to_bytes(feature)
        hash_values = self._get_hash_values(feature_bytes)
        
        return all(self.bit_array[hash_val] == 1 for hash_val in hash_values)
    
    def contains_detection(self, detection_data: dict) -> bool:
        """
        Check if detection data is likely in the Bloom filter.
        
        Parameters
        ----------
        detection_data : dict
            Detection data to check.
            
        Returns
        -------
        bool
            True if the detection is likely in the filter (may have false positives).
        """
        detection_bytes = self._detection_to_bytes(detection_data)
        hash_values = self._get_hash_values(detection_bytes)
        
        return all(self.bit_array[hash_val] == 1 for hash_val in hash_values)
    
    def contains_raw_bytes(self, data: bytes) -> bool:
        """
        Check if raw bytes data is likely in the Bloom filter.
        
        Parameters
        ----------
        data : bytes
            Raw bytes data to check.
            
        Returns
        -------
        bool
            True if the data is likely in the filter (may have false positives).
        """
        hash_values = self._get_hash_values(data)
        
        return all(self.bit_array[hash_val] == 1 for hash_val in hash_values)
    
    def get_false_positive_rate(self) -> float:
        """
        Calculate current false positive rate based on current element count.
        
        Returns
        -------
        float
            Current false positive rate.
        """
        if self.element_count == 0:
            return 0.0
        
        return (1 - math.exp(-self.hash_functions * self.element_count / self.bit_array_size)) ** self.hash_functions
    
    def get_memory_usage(self) -> int:
        """
        Get memory usage in bytes.
        
        Returns
        -------
        int
            Memory usage in bytes.
        """
        return self.bit_array.nbytes
    
    def get_fill_ratio(self) -> float:
        """
        Get the ratio of set bits in the filter.
        
        Returns
        -------
        float
            Ratio of set bits (0.0 to 1.0).
        """
        return np.sum(self.bit_array) / self.bit_array_size
    
    def clear(self) -> None:
        """Clear all elements from the Bloom filter."""
        self.bit_array.fill(0)
        self.element_count = 0
    
    def __len__(self) -> int:
        """Return the number of elements added to the filter."""
        return self.element_count
    
    def __contains__(self, item) -> bool:
        """
        Check if an item is in the Bloom filter.
        
        Parameters
        ----------
        item : Union[np.ndarray, dict, bytes]
            Item to check. Can be a feature vector, detection dict, or raw bytes.
            
        Returns
        -------
        bool
            True if the item is likely in the filter.
        """
        if isinstance(item, np.ndarray):
            return self.contains_feature(item)
        elif isinstance(item, dict):
            return self.contains_detection(item)
        elif isinstance(item, bytes):
            return self.contains_raw_bytes(item)
        else:
            raise TypeError(f"Unsupported item type: {type(item)}")


class TrackingBloomFilter:
    """
    A specialized Bloom filter for tracking applications with multiple filters.
    
    This class maintains separate Bloom filters for different types of tracking data
    and provides convenient methods for tracking-specific operations.
    """
    
    def __init__(self, expected_tracks: int = 1000, false_positive_rate: float = 0.01, feature_dim: int = 128):
        """
        Initialize tracking-specific Bloom filters.
        
        Parameters
        ----------
        expected_tracks : int
            Expected number of tracks to be stored.
        false_positive_rate : float
            Desired false positive rate for all filters.
        feature_dim : int
            Dimension of feature vectors (default: 128 for standard Deep SORT).
        """
        self.feature_filter = BloomFilter(
            expected_elements=expected_tracks * 10,  # Multiple features per track
            false_positive_rate=false_positive_rate,
            feature_dim=feature_dim  # Configurable feature dimension
        )
        
        self.detection_filter = BloomFilter(
            expected_elements=expected_tracks * 100,  # Many detections per track
            false_positive_rate=false_positive_rate
        )
        
        self.track_id_filter = BloomFilter(
            expected_elements=expected_tracks,
            false_positive_rate=false_positive_rate
        )
    
    def add_track_features(self, track_id: int, features: List[np.ndarray]) -> None:
        """
        Add features from a track to the Bloom filter.
        
        Parameters
        ----------
        track_id : int
            ID of the track.
        features : List[np.ndarray]
            List of feature vectors for the track.
        """
        if not features:
            return
            
        # Add track ID
        self.track_id_filter.add_raw_bytes(str(track_id).encode('utf-8'))
        
        # Add features
        for feature in features:
            if feature is not None and len(feature) > 0:
                self.feature_filter.add_feature(feature)
    
    def add_detection(self, detection_data: dict) -> None:
        """
        Add detection data to the Bloom filter.
        
        Parameters
        ----------
        detection_data : dict
            Detection data dictionary.
        """
        self.detection_filter.add_detection(detection_data)
    
    def is_track_known(self, track_id: int) -> bool:
        """
        Check if a track ID has been seen before.
        
        Parameters
        ----------
        track_id : int
            Track ID to check.
            
        Returns
        -------
        bool
            True if the track ID is likely known.
        """
        return self.track_id_filter.contains_raw_bytes(str(track_id).encode('utf-8'))
    
    def is_feature_known(self, feature: np.ndarray) -> bool:
        """
        Check if a feature vector has been seen before.
        
        Parameters
        ----------
        feature : np.ndarray
            Feature vector to check.
            
        Returns
        -------
        bool
            True if the feature is likely known.
        """
        return feature in self.feature_filter
    
    def is_detection_known(self, detection_data: dict) -> bool:
        """
        Check if detection data has been seen before.
        
        Parameters
        ----------
        detection_data : dict
            Detection data to check.
            
        Returns
        -------
        bool
            True if the detection is likely known.
        """
        return detection_data in self.detection_filter
    
    def get_stats(self) -> dict:
        """
        Get statistics about all Bloom filters.
        
        Returns
        -------
        dict
            Dictionary containing statistics for all filters.
        """
        return {
            'feature_filter': {
                'element_count': len(self.feature_filter),
                'false_positive_rate': self.feature_filter.get_false_positive_rate(),
                'memory_usage': self.feature_filter.get_memory_usage(),
                'fill_ratio': self.feature_filter.get_fill_ratio()
            },
            'detection_filter': {
                'element_count': len(self.detection_filter),
                'false_positive_rate': self.detection_filter.get_false_positive_rate(),
                'memory_usage': self.detection_filter.get_memory_usage(),
                'fill_ratio': self.detection_filter.get_fill_ratio()
            },
            'track_id_filter': {
                'element_count': len(self.track_id_filter),
                'false_positive_rate': self.track_id_filter.get_false_positive_rate(),
                'memory_usage': self.track_id_filter.get_memory_usage(),
                'fill_ratio': self.track_id_filter.get_fill_ratio()
            }
        }
