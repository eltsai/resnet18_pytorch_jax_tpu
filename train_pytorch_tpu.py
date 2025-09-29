"""
Main training script for PyTorch TPU training
Refactored version with modular components
"""
import torch_xla.distributed.xla_multiprocessing as xmp
from utils.logging_utils import load_config, save_config_copy
from utils.tpu_utils import set_seed
from training.pytorch_trainer_tpu import run_training


def _mp_fn(index):
    """Multi-processing function for TPU cores"""
    # Load configuration
    config = load_config('config/pytorch_tpu.yaml')
    
    # Set seed for reproducibility
    set_seed(config['logging']['seed'])
    
    # Save config copy (only on master process)
    import torch_xla.core.xla_model as xm
    if xm.is_master_ordinal():
        save_config_copy(config, config['paths']['log_dir'])
    
    # Run training
    run_training(config)


if __name__ == '__main__':
    """
    Entry point for TPU training
    Uses PJRT runtime with all available TPU cores
    """
    print("Starting PyTorch TPU Training...")
    xmp.spawn(_mp_fn, args=(), nprocs=None)
    print("Training completed!")