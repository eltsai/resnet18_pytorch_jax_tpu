"""
Optimized PyTorch TPU training script
Features: Mixed precision, model compilation, better data loading
"""
import torch_xla.distributed.xla_multiprocessing as xmp
from utils.logging_utils import load_config, save_config_copy
from utils.tpu_utils import set_seed
from training.pytorch_trainer_tpu_optimized import run_training


def _mp_fn(index):
    """Multi-processing function for TPU cores with optimizations"""
    # Load configuration
    config = load_config('config/pytorch_tpu_optimized.yaml')
    
    # Set seed for reproducibility
    set_seed(config['logging']['seed'])
    
    # Save config copy (only on master process)
    import torch_xla.core.xla_model as xm
    if xm.is_master_ordinal():
        save_config_copy(config, config['paths']['log_dir'])
    
    # Run optimized training
    run_training(config)


if __name__ == '__main__':
    """
    Entry point for optimized TPU training
    Uses PJRT runtime with all available TPU cores
    """
    print("Starting Optimized PyTorch TPU Training...")
    print("Features: Mixed Precision, Model Compilation, Optimized Data Loading")
    xmp.spawn(_mp_fn, args=(), nprocs=None)
    print("Optimized training completed!")