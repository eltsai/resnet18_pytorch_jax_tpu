"""
Optimized JAX utilities for high-performance TPU training
Best practices implementation
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple
from flax.training import train_state
from flax.training import orbax_utils
import orbax.checkpoint
import os


class TrainState(train_state.TrainState):
    """Custom TrainState with batch stats for BatchNorm"""
    batch_stats: Any
    epoch: int = 0
    best_acc: float = 0.0
    best_epoch: int = -1


def setup_jax_tpu() -> Tuple[int, bool]:
    """
    Setup JAX for TPU training with optimizations
    
    Returns:
        Tuple of (num_devices, is_tpu)
    """
    print("Initializing JAX for TPU...")
    
    # Basic JAX configurations for TPU
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('JAX_PLATFORMS', 'tpu,cpu')  # Prefer TPU, fallback to CPU
    
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
    print('OPTIMIZED JAX/FLAX TPU TRAINING')
    print('=' * 60)
    print(f"Devices: {jax.devices()}")
    print(f"Number of devices: {num_devices}")
    print(f"Device type: {'TPU' if is_tpu else 'CPU/GPU'}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"XLA flags: {os.environ.get('XLA_FLAGS', 'None set')}")
    print('=' * 60)


def save_checkpoint(state: TrainState, path: str):
    """Save model checkpoint with error handling and async support"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        
        # Try the new API first (Orbax >= 0.4.0)
        try:
            save_args = orbax_utils.save_args_from_target(state)
            # Use async save to avoid blocking main thread
            future = checkpointer.save(path, state, save_args=save_args)
            # Don't wait for completion to avoid blocking
            print(f"✓ Checkpoint save initiated for {path}")
            return future
        except TypeError:
            # Fallback to older API (Orbax < 0.4.0)
            try:
                save_args = orbax_utils.save_args_from_target(state)
                future = checkpointer.save(path, args=save_args, item=state)
                print(f"✓ Checkpoint save initiated for {path}")
                return future
            except TypeError:
                # If both fail, try the simplest approach
                future = checkpointer.save(path, state)
                print(f"✓ Checkpoint save initiated for {path}")
                return future
        
    except Exception as e:
        print(f"Warning: Failed to save checkpoint to {path}: {e}")
        return None


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
    return jax.random.PRNGKey(seed)


def get_checkpoint_paths(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Get full paths for best and last checkpoints
    
    Returns:
        Tuple of (best_checkpoint_path, last_checkpoint_path)
    """
    
    checkpoint_dir = config['paths']['checkpoint_dir']
    best_name = config['paths']['best_model_name']
    last_name = config['paths']['last_model_name']
    
    # Convert to absolute path to avoid Orbax warnings
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_path = os.path.join(checkpoint_dir, best_name)
    last_path = os.path.join(checkpoint_dir, last_name)
    
    return best_path, last_path


def optimize_for_tpu():
    """Apply TPU-specific optimizations"""
    
    # For now, let's skip XLA flags to avoid compatibility issues
    # Focus on JAX-level optimizations instead
    print("TPU optimizations: Using JAX default settings for maximum compatibility")
    
    # Clear any problematic XLA flags that might have been set
    if 'XLA_FLAGS' in os.environ:
        current_flags = os.environ['XLA_FLAGS']
        # Remove problematic flags
        problematic_flags = [
            '--xla_tpu_enable_latency_hiding_scheduler',
            '--xla_tpu_enable_async_collective_fusion',
            '--xla_tpu_enable_async_collective_fusion_fuse_all_gather', 
            '--xla_tpu_enable_async_collective_fusion_multiple_steps',
            '--xla_tpu_overlap_compute_collective_tc',
            '--xla_enable_async_all_gather'
        ]
        
        cleaned_flags = current_flags
        for flag in problematic_flags:
            if flag in cleaned_flags:
                # Remove the flag and its value
                import re
                pattern = rf'{re.escape(flag)}[^\s]*\s*'
                cleaned_flags = re.sub(pattern, '', cleaned_flags)
        
        os.environ['XLA_FLAGS'] = cleaned_flags.strip()
        if cleaned_flags.strip():
            print(f"Cleaned XLA_FLAGS: {cleaned_flags.strip()}")
        else:
            print("Cleared all XLA_FLAGS for compatibility")


# Apply optimizations when module is imported  
optimize_for_tpu()