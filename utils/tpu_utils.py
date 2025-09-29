"""
TPU utilities for PyTorch XLA training
"""
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import random
import numpy as np


def setup_tpu_device():
    """Setup TPU device and check if master process"""
    device = xm.xla_device()
    is_master = xm.is_master_ordinal()
    return device, is_master


def print_tpu_info(device, is_master=True):
    """Print TPU device information"""
    if is_master:
        print(f'Device: {device}')
        try:
            print(f'Available devices: {xm.get_xla_supported_devices()}')
            print(f'Real devices: {xm.xla_real_devices()}')
        except Exception as e:
            print(f'Could not get device info: {e}')


def set_seed(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_parallel_loader(dataloader, device):
    """Create XLA parallel loader for TPU"""
    return pl.MpDeviceLoader(dataloader, device)


def aggregate_metrics(metrics_dict):
    """
    Aggregate metrics across all TPU cores using mesh_reduce
    
    Args:
        metrics_dict: Dictionary with metric_name -> local_value
    
    Returns:
        Dictionary with aggregated values
    """
    aggregated = {}
    for name, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            aggregated[name] = xm.mesh_reduce(name, value, sum)
        else:
            aggregated[name] = value
    
    return aggregated


def sync_step():
    """Synchronization step for TPU"""
    xm.mark_step()


def save_checkpoint(state_dict, filepath):
    """Save checkpoint using XLA"""
    xm.save(state_dict, filepath)


def optimizer_step(optimizer):
    """XLA optimizer step with synchronization"""
    xm.optimizer_step(optimizer)