"""
JAX TPU utilities
"""
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Dict, Any, Tuple, Iterator
from flax.training import train_state
from flax.training import orbax_utils
import orbax.checkpoint

# Disable TensorFlow GPU usage to avoid conflicts with JAX TPU
tf.config.experimental.set_visible_devices([], "GPU")


class TrainState(train_state.TrainState):
    """Custom TrainState with batch stats for BatchNorm"""
    batch_stats: Any
    epoch: int = 0
    best_acc: float = 0.0
    best_epoch: int = -1


def setup_jax_tpu() -> Tuple[int, bool]:
    """
    Setup JAX for TPU training
    
    Returns:
        Tuple of (num_devices, is_tpu)
    """
    print("Initializing JAX for TPU...")
    
    # Check if running on TPU
    devices = jax.devices()
    num_devices = len(devices)
    # Check for TPU devices using the platform attribute
    is_tpu = devices[0].platform == 'tpu' if devices else False
    
    print(f"JAX devices: {devices}")
    print(f"Number of devices: {num_devices}")
    print(f"Device platform: {devices[0].platform if devices else 'None'}")
    print(f"Device kind: {devices[0].device_kind if devices else 'None'}")
    print(f"Device type: {'TPU' if is_tpu else 'CPU/GPU'}")
    
    if num_devices == 1 and is_tpu:
        print("Warning: Only 1 TPU device detected. Expected 8 for TPU v6e-8")
    elif not is_tpu:
        print("Warning: No TPU devices detected. Running on CPU/GPU")
    
    return num_devices, is_tpu


def print_tpu_info(num_devices: int, is_tpu: bool):
    """Print TPU information"""
    print('=' * 60)
    print('JAX/FLAX TPU TRAINING INFORMATION')
    print('=' * 60)
    print(f"Devices: {jax.devices()}")
    print(f"Number of devices: {num_devices}")
    print(f"Device type: {'TPU' if is_tpu else 'CPU/GPU'}")
    print('=' * 60)


def normalize_image(image, label, mean, std):
    """Normalize CIFAR-10 images"""
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)
    image = (image - mean) / std
    return image, label


def augment_image(image, label, config):
    """Apply data augmentation for training"""
    # Random horizontal flip
    if config['transforms']['train'].get('random_horizontal_flip', False):
        image = tf.image.random_flip_left_right(image)
    
    # Random crop with padding
    if 'random_crop' in config['transforms']['train']:
        crop_config = config['transforms']['train']['random_crop']
        padding = crop_config['padding']
        size = crop_config['size']
        
        # Pad and then random crop
        padded_size = size + 2 * padding
        image = tf.image.resize_with_crop_or_pad(image, padded_size, padded_size)
        image = tf.image.random_crop(image, [size, size, 3])
    
    return image, label


def create_datasets(config: Dict[str, Any]) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create CIFAR-10 datasets using TensorFlow Datasets with optimizations"""
    
    per_device_batch_size = config['training']['per_device_batch_size']
    num_devices = len(jax.devices())
    total_batch_size = per_device_batch_size * num_devices
    
    print(f"Creating datasets: per_device_batch_size={per_device_batch_size}, num_devices={num_devices}, total_batch_size={total_batch_size}")
    
    # Load CIFAR-10 with better caching
    ds_train, ds_test = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        try_gcs=False,  # Avoid GCS if not needed
        download=False   # Assume already downloaded
    )
    
    # Get normalization parameters
    train_mean = config['transforms']['train']['normalize']['mean']
    train_std = config['transforms']['train']['normalize']['std']
    test_mean = config['transforms']['test']['normalize']['mean']
    test_std = config['transforms']['test']['normalize']['std']
    
    # Optimize training pipeline
    ds_train = ds_train.cache()  # Cache after loading to avoid re-reading
    ds_train = ds_train.shuffle(config['data']['shuffle_buffer_size'], reshuffle_each_iteration=True)
    ds_train = ds_train.repeat()  # Infinite repeat to avoid recreating iterator
    ds_train = ds_train.map(
        lambda img, lbl: augment_image(img, lbl, config),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.map(
        lambda img, lbl: normalize_image(img, lbl, train_mean, train_std),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.batch(total_batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)  # Use AUTOTUNE for better performance
    
    # Optimize test pipeline
    test_batch_size = total_batch_size * config['data']['test_batch_multiplier']
    ds_test = ds_test.cache()  # Cache test set too
    ds_test = ds_test.map(
        lambda img, lbl: normalize_image(img, lbl, test_mean, test_std),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_test = ds_test.batch(test_batch_size, drop_remainder=True)
    ds_test = ds_test.prefetch(buffer_size=1)  # Smaller buffer for test
    
    return ds_train, ds_test


def create_data_iterators(ds_train: tf.data.Dataset, ds_test: tf.data.Dataset) -> Tuple[Iterator, Iterator]:
    """Create JAX-compatible data iterators"""
    train_iter = iter(tfds.as_numpy(ds_train))
    test_iter = iter(tfds.as_numpy(ds_test))
    return train_iter, test_iter


def save_checkpoint(state: TrainState, path: str):
    """Save model checkpoint"""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(path, args=save_args, item=state)


def aggregate_metrics_across_devices(metrics: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    """
    Aggregate metrics across devices for multi-device training
    
    Args:
        metrics: Dictionary of metrics (each value should be array from devices)
        
    Returns:
        Dictionary of aggregated metrics as Python floats
    """
    aggregated = {}
    for key, value in metrics.items():
        if isinstance(value, jnp.ndarray):
            aggregated[key] = float(jnp.mean(value))
        else:
            aggregated[key] = float(value)
    
    return aggregated


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    return jax.random.PRNGKey(seed)


def get_checkpoint_paths(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Get full paths for best and last checkpoints
    
    Returns:
        Tuple of (best_checkpoint_path, last_checkpoint_path)
    """
    import os
    
    checkpoint_dir = config['paths']['checkpoint_dir']
    best_name = config['paths']['best_model_name']
    last_name = config['paths']['last_model_name']
    
    best_path = os.path.join(checkpoint_dir, best_name)
    last_path = os.path.join(checkpoint_dir, last_name)
    
    return best_path, last_path