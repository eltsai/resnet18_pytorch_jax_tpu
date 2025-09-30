"""
Logging utilities for training
"""
import os
import yaml
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create necessary directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    return config


def get_checkpoint_paths(config: Dict[str, Any]) -> tuple:
    """
    Get full paths for best and last checkpoints
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (best_checkpoint_path, last_checkpoint_path)
    """
    checkpoint_dir = config['paths']['checkpoint_dir']
    best_name = config['paths']['best_model_name']
    last_name = config['paths']['last_model_name']
    
    best_path = os.path.join(checkpoint_dir, best_name)
    last_path = os.path.join(checkpoint_dir, last_name)
    
    return best_path, last_path


def log_training_info(config: Dict[str, Any], is_master: bool = True):
    """
    Log training configuration and setup information
    
    Args:
        config: Configuration dictionary
        is_master: Whether this is the master process
    """
    if not is_master:
        return
    
    print('=' * 60)
    
    # Detect framework based on config structure
    if 'batch_size' in config['training']:
        # PyTorch configuration
        print('PYTORCH TPU TRAINING CONFIGURATION')
        print('=' * 60)
        training_cfg = config['training']
        per_core_batch_size = training_cfg['batch_size']
        total_cores = config['tpu']['total_cores']
        total_batch_size = per_core_batch_size * total_cores
        core_label = "Per-Core Batch Size"
        cores_label = "TPU Cores"
    else:
        # JAX configuration  
        print('JAX/FLAX TPU TRAINING CONFIGURATION')
        print('=' * 60)
        training_cfg = config['training']
        per_core_batch_size = training_cfg['per_device_batch_size']
        total_cores = config['tpu']['total_devices']
        total_batch_size = per_core_batch_size * total_cores
        core_label = "Per-Device Batch Size"
        cores_label = "TPU Devices"
    
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['data']['dataset']}")
    print(f"Total epochs: {training_cfg['total_epochs']}")
    print(f"{core_label}: {per_core_batch_size}")
    print(f"Total Effective Batch Size: {total_batch_size}")
    print(f"Base Learning Rate: {training_cfg['base_lr']}")
    print(f"Weight Decay: {training_cfg['weight_decay']}")
    print(f"Warmup Epochs: {training_cfg['warmup_epochs']}")
    print(f"Epochs per testing: {training_cfg['epochs_per_testing']}")
    print(f"{cores_label}: {total_cores}")
    print('=' * 60)


def log_epoch_info(epoch: int, total_epochs: int, lr: float, 
                   is_master: bool = True, is_first_epoch: bool = False):
    """
    Log epoch information
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        lr: Current learning rate
        is_master: Whether this is the master process
        is_first_epoch: Whether this is the first epoch
    """
    if not is_master:
        return
    
    print(f'\n--- Epoch {epoch + 1}/{total_epochs} (LR: {lr:.6f}) ---')
    if is_first_epoch:
        print("⚠️  First epoch may take longer due to XLA compilation...")


def log_metrics(epoch: int, train_loss: float, train_acc: float, 
                test_loss: float = None, test_acc: float = None,
                epoch_time: float = None, is_master: bool = True):
    """
    Log training and test metrics
    
    Args:
        epoch: Current epoch (0-indexed)
        train_loss: Training loss
        train_acc: Training accuracy
        test_loss: Test loss (optional)
        test_acc: Test accuracy (optional)
        epoch_time: Time taken for epoch (optional)
        is_master: Whether this is the master process
    """
    if not is_master:
        return
    
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%')
    
    if test_loss is not None and test_acc is not None:
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%')
    
    if epoch_time is not None:
        print(f'Epoch Time: {epoch_time:.2f}s')


def log_best_checkpoint(epoch: int, accuracy: float, is_master: bool = True):
    """
    Log when a new best checkpoint is saved
    
    Args:
        epoch: Epoch number (1-indexed)
        accuracy: Best accuracy achieved
        is_master: Whether this is the master process
    """
    if not is_master:
        return
    
    print(f"-> New best checkpoint saved at epoch {epoch} with Acc: {accuracy * 100:.2f}%")


def log_final_results(best_acc: float, best_epoch: int, total_time: float, 
                     is_master: bool = True):
    """
    Log final training results
    
    Args:
        best_acc: Best accuracy achieved
        best_epoch: Epoch where best accuracy was achieved
        total_time: Total training time in seconds
        is_master: Whether this is the master process
    """
    if not is_master:
        return
    
    print('\n' + '=' * 30)
    print(f'Best Acc: {best_acc * 100:.2f}% (Epoch {best_epoch})')
    print('=' * 60)
    print(f'Total time consumed: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)')
    print('=' * 60)


def save_config_copy(config: Dict[str, Any], log_dir: str):
    """
    Save a copy of the configuration used for training
    
    Args:
        config: Configuration dictionary
        log_dir: Directory to save config copy
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_copy_path = os.path.join(log_dir, f'config_used_{timestamp}.yaml')
    
    with open(config_copy_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to: {config_copy_path}")
    return config_copy_path


def save_training_stats(train_loss_history: List[float], train_acc_history: List[float], 
                       test_loss_history: List[float], test_acc_history: List[float], 
                       best_acc: float, best_epoch: int, total_time: float,
                       config: Dict[str, Any]):
    """
    Save training statistics to JSON file
    
    Args:
        train_loss_history: List of training losses
        train_acc_history: List of training accuracies
        test_loss_history: List of test losses (may contain NaN values)
        test_acc_history: List of test accuracies (may contain NaN values)
        best_acc: Best accuracy achieved
        best_epoch: Epoch where best accuracy was achieved
        total_time: Total training time in seconds
        config: Configuration dictionary
    """
    stats = {
        'training': {
            'loss_history': [float(x) for x in train_loss_history],
            'accuracy_history': [float(x) for x in train_acc_history]
        },
        'testing': {
            'loss_history': [float(x) if not np.isnan(x) else None for x in test_loss_history],
            'accuracy_history': [float(x) if not np.isnan(x) else None for x in test_acc_history]
        },
        'best_results': {
            'best_accuracy': float(best_acc),
            'best_epoch': int(best_epoch),
            'total_training_time_seconds': float(total_time)
        },
        'config': {
            'total_epochs': config['training']['total_epochs'],
            'epochs_per_testing': config['training']['epochs_per_testing'],
            'per_core_batch_size': config['training'].get('per_device_batch_size', config['training'].get('batch_size')),
            'base_lr': config['training']['base_lr'],
            'weight_decay': config['training']['weight_decay'],
            'warmup_epochs': config['training']['warmup_epochs']
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    log_dir = config['paths']['log_dir']
    stats_path = os.path.join(log_dir, 'training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Training statistics saved to: {stats_path}")
    return stats_path