"""
Main training script for JAX/Flax TPU training
Modular version with separate components
"""
from utils.logging_utils import load_config, save_config_copy
from utils.jax_utils_optimized import set_seed
from training.jax_trainer_tpu import run_training


def main():
    """Main training function"""
    # Load configuration
    config = load_config('config/jax_tpu.yaml')
    
    # Set seed for reproducibility
    set_seed(config['logging']['seed'])
    
    # Save config copy
    save_config_copy(config, config['paths']['log_dir'])
    
    # Run training
    run_training(config)


if __name__ == '__main__':
    """
    Entry point for JAX TPU training
    Uses pure JAX without PyTorch dependencies
    """
    print("Starting JAX/Flax TPU Training...")
    main()
    print("Training completed!")