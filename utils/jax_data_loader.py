"""
Pure JAX/NumPy data loader for CIFAR-10
No TensorFlow dependencies - much faster for JAX/TPU
"""
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple, Iterator, Generator
import functools


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
    """Normalize data with given mean and std"""
    data = data / 255.0  # Convert to [0, 1]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
    return (data - mean) / std


@functools.partial(jax.jit, static_argnames=['training'])
def augment_batch(key: jax.random.PRNGKey, batch_data: jnp.ndarray, training: bool = True) -> jnp.ndarray:
    """JAX-native data augmentation"""
    if not training:
        return batch_data
    
    batch_size = batch_data.shape[0]
    keys = jax.random.split(key, batch_size)
    
    def augment_single(key, image):
        # Random horizontal flip
        flip_key, crop_key = jax.random.split(key)
        
        # Flip with 50% probability
        should_flip = jax.random.bernoulli(flip_key, 0.5)
        image = jnp.where(should_flip, jnp.fliplr(image), image)
        
        # Random crop with padding
        # Pad image to 40x40, then crop back to 32x32
        padded = jnp.pad(image, ((4, 4), (4, 4), (0, 0)), mode='reflect')
        
        # Random crop coordinates
        crop_h = jax.random.randint(crop_key, (), 0, 9)  # 0 to 8 (40-32)
        crop_w = jax.random.randint(crop_key, (), 0, 9)
        
        cropped = jax.lax.dynamic_slice(padded, (crop_h, crop_w, 0), (32, 32, 3))
        
        return cropped
    
    return jax.vmap(augment_single)(keys, batch_data)


class JAXDataLoader:
    """Pure JAX data loader - no TensorFlow dependencies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config['training']['per_device_batch_size'] * len(jax.devices())
        
        # Load data
        data_dir = os.path.join(config['data']['data_path'], 'cifar-10-batches-py')
        self.train_data, self.train_labels, self.test_data, self.test_labels = load_cifar10(data_dir)
        
        # Normalize data
        train_mean = config['transforms']['train']['normalize']['mean']
        train_std = config['transforms']['train']['normalize']['std']
        test_mean = config['transforms']['test']['normalize']['mean']  
        test_std = config['transforms']['test']['normalize']['std']
        
        self.train_data = normalize_data(self.train_data, train_mean, train_std)
        self.test_data = normalize_data(self.test_data, test_mean, test_std)
        
        # Convert to JAX arrays and move to devices
        self.train_data = jnp.array(self.train_data)
        self.train_labels = jnp.array(self.train_labels)
        self.test_data = jnp.array(self.test_data)
        self.test_labels = jnp.array(self.test_labels)
        
        self.num_train_samples = len(self.train_data)
        self.num_test_samples = len(self.test_data)
        self.steps_per_epoch = self.num_train_samples // self.batch_size
        
        print(f"JAX DataLoader initialized:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Steps per epoch: {self.steps_per_epoch}")
        print(f"  Train samples: {self.num_train_samples}")
        print(f"  Test samples: {self.num_test_samples}")
    
    def get_train_batches(self, key: jax.random.PRNGKey, epoch: int) -> Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]:
        """Generate training batches for one epoch"""
        # Shuffle data for this epoch
        epoch_key = jax.random.fold_in(key, epoch)
        indices = jax.random.permutation(epoch_key, self.num_train_samples)
        
        # Generate batches
        for step in range(self.steps_per_epoch):
            start_idx = step * self.batch_size
            end_idx = start_idx + self.batch_size
            
            batch_indices = indices[start_idx:end_idx]
            batch_data = self.train_data[batch_indices]
            batch_labels = self.train_labels[batch_indices]
            
            # Apply augmentation
            aug_key = jax.random.fold_in(epoch_key, step)
            batch_data = augment_batch(aug_key, batch_data, training=True)
            
            yield batch_data, batch_labels
    
    def get_test_batches(self) -> Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]:
        """Generate test batches"""
        test_batch_size = self.batch_size * self.config['data']['test_batch_multiplier']
        num_test_batches = self.num_test_samples // test_batch_size
        
        for step in range(num_test_batches):
            start_idx = step * test_batch_size
            end_idx = start_idx + test_batch_size
            
            batch_data = self.test_data[start_idx:end_idx]
            batch_labels = self.test_labels[start_idx:end_idx]
            
            yield batch_data, batch_labels


def create_jax_data_loader(config: Dict[str, Any]) -> JAXDataLoader:
    """Create JAX-native data loader"""
    return JAXDataLoader(config)