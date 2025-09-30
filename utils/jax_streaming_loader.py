"""
Optimized JAX data loader with streaming and async preprocessing
Best practices for JAX/TPU training performance
"""
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import threading
import queue
from typing import Dict, Any, Tuple, Iterator, Generator
import functools
from concurrent.futures import ThreadPoolExecutor
import time


def load_cifar10_batch(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single CIFAR-10 batch file"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    data = batch[b'data']
    labels = batch[b'labels']
    
    # Reshape from (10000, 3072) to (10000, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return data.astype(np.float32), np.array(labels, dtype=np.int32)


def load_cifar10(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load complete CIFAR-10 dataset"""
    
    # Load training batches
    train_data = []
    train_labels = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(batch_file)
        train_data.append(data)
        train_labels.append(labels)
    
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    # Load test batch
    test_file = os.path.join(data_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_file)
    
    print(f"Loaded CIFAR-10: train_data.shape={train_data.shape}, test_data.shape={test_data.shape}")
    
    return train_data, train_labels, test_data, test_labels


def normalize_data(data: np.ndarray, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> np.ndarray:
    """Normalize data with given mean and std - CPU operation"""
    data = data / 255.0  # Convert to [0, 1]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
    return (data - mean) / std


def augment_image_cpu(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """CPU-based data augmentation for single image - much faster than JAX"""
    
    # Random horizontal flip
    if rng.random() < 0.5:
        image = np.fliplr(image)
    
    # Random crop with padding
    # Pad image to 40x40, then crop back to 32x32
    padded = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='reflect')
    
    # Random crop coordinates
    crop_h = rng.integers(0, 9)  # 0 to 8 (40-32)
    crop_w = rng.integers(0, 9)
    
    cropped = padded[crop_h:crop_h+32, crop_w:crop_w+32, :]
    
    return cropped


def augment_batch_cpu(batch_data: np.ndarray, seed: int) -> np.ndarray:
    """CPU-based batch augmentation - parallel processing"""
    rng = np.random.default_rng(seed)
    
    # Create seeds for each sample in the batch
    seeds = rng.integers(0, 2**31, size=batch_data.shape[0])
    
    # Process each image
    augmented = np.zeros_like(batch_data)
    for i, (image, img_seed) in enumerate(zip(batch_data, seeds)):
        img_rng = np.random.default_rng(img_seed)
        augmented[i] = augment_image_cpu(image, img_rng)
    
    return augmented


class StreamingDataLoader:
    """Streaming data loader with async preprocessing - JAX best practices"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.global_batch_size = config['training']['per_device_batch_size'] * len(jax.devices())
        self.num_devices = len(jax.devices())
        self.per_device_batch_size = config['training']['per_device_batch_size']
        
        # Load data once - keep on CPU
        data_dir = os.path.join(config['data']['data_path'], 'cifar-10-batches-py')
        self.train_data, self.train_labels, self.test_data, self.test_labels = load_cifar10(data_dir)
        
        # Normalize data on CPU
        train_mean = config['transforms']['train']['normalize']['mean']
        train_std = config['transforms']['train']['normalize']['std']
        test_mean = config['transforms']['test']['normalize']['mean']  
        test_std = config['transforms']['test']['normalize']['std']
        
        self.train_data = normalize_data(self.train_data, train_mean, train_std)
        self.test_data = normalize_data(self.test_data, test_mean, test_std)
        
        self.num_train_samples = len(self.train_data)
        self.num_test_samples = len(self.test_data)
        self.steps_per_epoch = self.num_train_samples // self.global_batch_size
        
        # Threading for async data loading
        self.num_workers = config['data'].get('num_workers', 4)
        self.prefetch_buffer = config['data'].get('prefetch_buffer', 2)
        
        print(f"Streaming DataLoader initialized:")
        print(f"  Global batch size: {self.global_batch_size}")
        print(f"  Per-device batch size: {self.per_device_batch_size}")
        print(f"  Num devices: {self.num_devices}")
        print(f"  Steps per epoch: {self.steps_per_epoch}")
        print(f"  Train samples: {self.num_train_samples}")
        print(f"  Test samples: {self.num_test_samples}")
        print(f"  Num workers: {self.num_workers}")
    
    def _prepare_batch_worker(self, batch_data: np.ndarray, batch_labels: np.ndarray, 
                            seed: int, augment: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Worker function for preparing a single batch"""
        
        if augment:
            # CPU augmentation
            batch_data = augment_batch_cpu(batch_data, seed)
        
        # Convert to JAX arrays
        batch_data = jnp.array(batch_data)
        batch_labels = jnp.array(batch_labels)
        
        # Reshape for multi-device if needed
        if self.num_devices > 1:
            batch_data = batch_data.reshape(self.num_devices, self.per_device_batch_size, *batch_data.shape[1:])
            batch_labels = batch_labels.reshape(self.num_devices, self.per_device_batch_size)
        
        return batch_data, batch_labels
    
    def get_train_batches(self, epoch: int, seed: int = 42) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get training batches with async preprocessing"""
        
        # Create epoch-specific permutation
        rng = np.random.default_rng(seed + epoch)
        indices = rng.permutation(self.num_train_samples)
        
        # Use ThreadPoolExecutor for async batch preparation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit first few batches
            future_batches = {}
            
            for step in range(min(self.prefetch_buffer, self.steps_per_epoch)):
                start_idx = step * self.global_batch_size
                end_idx = start_idx + self.global_batch_size
                
                batch_indices = indices[start_idx:end_idx]
                batch_data = self.train_data[batch_indices].copy()  # Copy to avoid memory issues
                batch_labels = self.train_labels[batch_indices].copy()
                
                batch_seed = seed + epoch * 10000 + step
                future = executor.submit(
                    self._prepare_batch_worker, 
                    batch_data, batch_labels, batch_seed, True
                )
                future_batches[step] = future
            
            # Yield batches and submit new ones
            for step in range(self.steps_per_epoch):
                # Get the prepared batch
                if step in future_batches:
                    batch_data, batch_labels = future_batches[step].result()
                    del future_batches[step]
                else:
                    # Fallback - prepare synchronously
                    start_idx = step * self.global_batch_size
                    end_idx = start_idx + self.global_batch_size
                    batch_indices = indices[start_idx:end_idx]
                    batch_data = self.train_data[batch_indices].copy()
                    batch_labels = self.train_labels[batch_indices].copy()
                    batch_seed = seed + epoch * 10000 + step
                    batch_data, batch_labels = self._prepare_batch_worker(
                        batch_data, batch_labels, batch_seed, True
                    )
                
                # Submit next batch if available
                next_step = step + self.prefetch_buffer
                if next_step < self.steps_per_epoch and next_step not in future_batches:
                    start_idx = next_step * self.global_batch_size
                    end_idx = start_idx + self.global_batch_size
                    batch_indices = indices[start_idx:end_idx]
                    next_batch_data = self.train_data[batch_indices].copy()
                    next_batch_labels = self.train_labels[batch_indices].copy()
                    batch_seed = seed + epoch * 10000 + next_step
                    future = executor.submit(
                        self._prepare_batch_worker,
                        next_batch_data, next_batch_labels, batch_seed, True
                    )
                    future_batches[next_step] = future
                
                yield batch_data, batch_labels
    
    def get_test_batches(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get test batches - no augmentation, simpler processing"""
        test_batch_size = self.global_batch_size * self.config['data']['test_batch_multiplier']
        num_test_batches = self.num_test_samples // test_batch_size
        
        for step in range(num_test_batches):
            start_idx = step * test_batch_size
            end_idx = start_idx + test_batch_size
            
            batch_data = self.test_data[start_idx:end_idx].copy()
            batch_labels = self.test_labels[start_idx:end_idx].copy()
            
            # Convert to JAX arrays
            batch_data = jnp.array(batch_data)
            batch_labels = jnp.array(batch_labels)
            
            # Reshape for multi-device if needed
            if self.num_devices > 1:
                per_device_test_batch_size = test_batch_size // self.num_devices
                batch_data = batch_data.reshape(self.num_devices, per_device_test_batch_size, *batch_data.shape[1:])
                batch_labels = batch_labels.reshape(self.num_devices, per_device_test_batch_size)
            
            yield batch_data, batch_labels


def create_streaming_data_loader(config: Dict[str, Any]) -> StreamingDataLoader:
    """Create optimized streaming data loader"""
    return StreamingDataLoader(config)