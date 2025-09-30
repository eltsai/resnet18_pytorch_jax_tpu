"""
Optimized PyTorch TPU Trainer for ResNet18 on CIFAR-10
Key optimizations:
1. Mixed precision training
2. Compiled models
3. Optimized data loading
4. Better memory management
5. Asynchronous operations
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List

# XLA imports
import torch_xla.core.xla_model as xm
import torch_xla.amp as xla_amp
import torch_xla.debug.metrics as met

from models.pytorch.resnet18 import resnet18
from utils.tpu_utils import (
    setup_tpu_device, print_tpu_info, create_parallel_loader,
    aggregate_metrics, sync_step, save_checkpoint, optimizer_step
)
from utils.logging_utils import (
    log_training_info, log_epoch_info, log_metrics, 
    log_best_checkpoint, log_final_results, get_checkpoint_paths
)
from utils.plotting import create_training_plots


class OptimizedTPUTrainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Optimized TPU Trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device, self.is_master = setup_tpu_device()
        
        # Mixed precision support with compatibility check
        self.use_amp = config.get('training', {}).get('mixed_precision', True)
        if self.use_amp:
            try:
                self.scaler = xla_amp.GradScaler()
                # Test autocast compatibility
                with xla_amp.autocast(self.device):
                    pass
                print("✓ Mixed precision (AMP) enabled")
            except Exception as e:
                print(f"Warning: Mixed precision not available: {e}")
                print("Falling back to FP32 training")
                self.use_amp = False
        
        # Training history
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []
        
        # Training state
        self.best_acc = 0.0
        self.best_epoch = -1
        self.current_epoch = 0
        
        # Performance tracking
        self.step_count = 0
        self.compile_metrics = []
        
        # Get paths
        self.best_ckpt_path, self.last_ckpt_path = get_checkpoint_paths(config)
        
    def setup_data_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create optimized data transforms"""
        transform_cfg = self.config['transforms']
        
        # Training transforms with optimizations
        train_transforms = []
        
        # Use more efficient transforms
        if 'random_crop' in transform_cfg['train']:
            crop_cfg = transform_cfg['train']['random_crop']
            train_transforms.append(
                transforms.RandomCrop(
                    crop_cfg['size'], 
                    padding=crop_cfg['padding'],
                    padding_mode=crop_cfg['padding_mode']
                )
            )
        
        if transform_cfg['train'].get('random_horizontal_flip'):
            train_transforms.append(transforms.RandomHorizontalFlip())
        
        # Convert to tensor first
        train_transforms.append(transforms.ToTensor())
        
        # Normalize (this should be done on TPU for efficiency)
        norm_cfg = transform_cfg['train']['normalize']
        train_transforms.append(
            transforms.Normalize(norm_cfg['mean'], norm_cfg['std'])
        )
        
        # Test transforms
        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(
                transform_cfg['test']['normalize']['mean'],
                transform_cfg['test']['normalize']['std']
            )
        ]
        
        return transforms.Compose(train_transforms), transforms.Compose(test_transforms)
    
    def setup_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Setup optimized data loaders"""
        transform_train, transform_test = self.setup_data_transforms()
        
        data_cfg = self.config['data']
        training_cfg = self.config['training']
        
        # Use optimized data loading parameters
        num_workers = min(data_cfg['num_workers'], 8)  # Too many workers can hurt TPU performance
        
        # Datasets
        trainset = torchvision.datasets.CIFAR10(
            root=data_cfg['root'], 
            train=True, 
            download=self.is_master, 
            transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_cfg['root'], 
            train=False, 
            download=self.is_master, 
            transform=transform_test
        )
        
        # Optimized data loaders
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=training_cfg['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=False,  # Not needed for TPU
            persistent_workers=True if num_workers > 0 else False
        )
        
        test_batch_size = training_cfg['batch_size'] * data_cfg['test_batch_multiplier']
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return trainloader, testloader
    
    def setup_model_and_optimizer(self) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Setup optimized model, criterion, optimizer and scheduler"""
        model_cfg = self.config['model']
        training_cfg = self.config['training']
        
        # Model
        model = resnet18(num_classes=model_cfg['num_classes']).to(self.device)
        
        # Compile model for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile') and training_cfg.get('compile_model', True):
            try:
                model = torch.compile(model, backend='openxla')
                print("✓ Model compiled with OpenXLA backend")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
        
        # Criterion
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer with optimized settings
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_cfg['base_lr'],
            momentum=training_cfg['momentum'],
            weight_decay=training_cfg['weight_decay'],
            nesterov=training_cfg['nesterov']
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_cfg['total_epochs'],
            eta_min=training_cfg['scheduler_eta_min']
        )
        
        optimizer.param_groups[0]['initial_lr'] = training_cfg['base_lr']
        
        return model, criterion, optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, criterion: nn.Module, 
                   optimizer: optim.Optimizer, trainloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Optimized training for one epoch"""
        model.train()
        total_train_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0
        
        train_device_loader = create_parallel_loader(trainloader, self.device)
        
        for batch_idx, (inputs, targets) in enumerate(train_device_loader):
            optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                # Mixed precision training with error handling
                try:
                    with xla_amp.autocast(self.device):
                        logits = model(inputs)
                        loss = criterion(logits, targets)
                except TypeError:
                    # Fallback for older XLA versions
                    with xla_amp.autocast():
                        logits = model(inputs)
                        loss = criterion(logits, targets)
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Regular training
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer_step(optimizer)
            
            # Collect metrics locally
            with torch.no_grad():
                total_train_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                train_correct_predictions += predicted.eq(targets).sum().item()
                train_total_samples += targets.size(0)
            
            self.step_count += 1
            
            # Periodic sync for better performance
            if batch_idx % 10 == 0:
                sync_step()
        
        # Final sync
        sync_step()
        
        # Aggregate metrics across cores
        metrics = aggregate_metrics({
            'train_loss_sum': total_train_loss,
            'train_correct_sum': train_correct_predictions,
            'train_samples_sum': train_total_samples
        })
        
        avg_train_loss = metrics['train_loss_sum'] / metrics['train_samples_sum']
        train_acc = metrics['train_correct_sum'] / metrics['train_samples_sum']
        
        return avg_train_loss, train_acc
    
    @torch.no_grad()
    def evaluate(self, model: nn.Module, criterion: nn.Module, 
                testloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Optimized evaluation"""
        model.eval()
        total_test_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        test_device_loader = create_parallel_loader(testloader, self.device)
        
        for batch_idx, (inputs, targets) in enumerate(test_device_loader):
            if self.use_amp:
                try:
                    with xla_amp.autocast(self.device):
                        logits = model(inputs)
                        loss = criterion(logits, targets)
                except TypeError:
                    # Fallback for older XLA versions
                    with xla_amp.autocast():
                        logits = model(inputs)
                        loss = criterion(logits, targets)
            else:
                logits = model(inputs)
                loss = criterion(logits, targets)
            
            total_test_loss += loss.item() * inputs.size(0)
            _, predicted = logits.max(1)
            total_samples += targets.size(0)
            correct_predictions += predicted.eq(targets).sum().item()
            
            # Periodic sync
            if batch_idx % 10 == 0:
                sync_step()
        
        # Final sync
        sync_step()
        
        # Aggregate metrics across cores
        metrics = aggregate_metrics({
            'test_loss_sum': total_test_loss,
            'test_correct_sum': correct_predictions,
            'test_samples_sum': total_samples
        })
        
        avg_test_loss = metrics['test_loss_sum'] / metrics['test_samples_sum']
        test_acc = metrics['test_correct_sum'] / metrics['test_samples_sum']
        
        return avg_test_loss, test_acc
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: optim.lr_scheduler._LRScheduler, epoch: int, 
                       is_best: bool = False):
        """Optimized checkpoint saving"""
        if not self.is_master:
            return
        
        # Get model state without compilation wrapper
        if hasattr(model, '_orig_mod'):
            model_state = model._orig_mod.state_dict()
        else:
            model_state = model.state_dict()
        
        state = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': self.best_acc,
            'step_count': self.step_count
        }
        
        if is_best:
            save_checkpoint(state, self.best_ckpt_path)
        else:
            save_checkpoint(state, self.last_ckpt_path)
    
    def print_performance_metrics(self):
        """Print XLA performance metrics"""
        if self.is_master:
            print("\n" + "="*60)
            print("XLA Performance Metrics:")
            print("="*60)
            try:
                print(met.metrics_report())
            except Exception as e:
                print(f"Could not get metrics: {e}")
            print("="*60)
    
    def train(self):
        """Optimized main training loop"""
        # Setup
        print_tpu_info(self.device, self.is_master)
        log_training_info(self.config, self.is_master)
        
        trainloader, testloader = self.setup_data_loaders()
        model, criterion, optimizer, scheduler = self.setup_model_and_optimizer()
        
        training_cfg = self.config['training']
        start_time = datetime.now()
        
        # Training loop
        for epoch in range(training_cfg['total_epochs']):
            epoch_start_time = time.time()
            
            # Warmup logic (optimized)
            if epoch < training_cfg['warmup_epochs']:
                lr_scale = (epoch + 1) / training_cfg['warmup_epochs']
                current_lr = training_cfg['base_lr'] * lr_scale
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # Log epoch info
            log_epoch_info(epoch, training_cfg['total_epochs'], current_lr, 
                          self.is_master, epoch == 0)
            
            # Training
            avg_train_loss, train_acc = self.train_epoch(model, criterion, optimizer, trainloader)
            
            if self.is_master:
                self.train_loss_history.append(avg_train_loss)
                self.train_acc_history.append(train_acc)
            
            # Step scheduler
            if epoch >= training_cfg['warmup_epochs'] - 1:
                scheduler.step()
            
            # Evaluation and checkpointing
            should_evaluate = ((epoch + 1) % training_cfg['epochs_per_testing'] == 0 or 
                             (epoch + 1) == training_cfg['total_epochs'])
            
            if should_evaluate:
                avg_test_loss, test_acc = self.evaluate(model, criterion, testloader)
                
                if self.is_master:
                    self.test_loss_history.append(avg_test_loss)
                    self.test_acc_history.append(test_acc)
                    
                    # Check for best model
                    if test_acc > self.best_acc:
                        self.best_acc = test_acc
                        self.best_epoch = epoch + 1
                        self.save_checkpoint(model, optimizer, scheduler, 
                                           self.best_epoch, is_best=True)
                        log_best_checkpoint(self.best_epoch, self.best_acc, self.is_master)
                
                sync_step()
                
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                log_metrics(epoch, avg_train_loss, train_acc, 
                           avg_test_loss, test_acc, epoch_time, self.is_master)
            else:
                if self.is_master:
                    self.test_loss_history.append(np.nan)
                    self.test_acc_history.append(np.nan)
                
                epoch_time = time.time() - epoch_start_time
                log_metrics(epoch, avg_train_loss, train_acc, 
                           epoch_time=epoch_time, is_master=self.is_master)
            
            # Print performance metrics periodically
            if epoch % 20 == 0 and self.is_master:
                self.print_performance_metrics()
            
            self.current_epoch = epoch
        
        # Final logging and saving
        if self.is_master:
            # Save final checkpoint
            self.save_checkpoint(model, optimizer, scheduler, self.current_epoch + 1)
            
            # Log final results
            total_time = (datetime.now() - start_time).total_seconds()
            log_final_results(self.best_acc, self.best_epoch, total_time, self.is_master)
            
            # Final performance metrics
            self.print_performance_metrics()
            
            # Save training statistics to JSON
            from utils.logging_utils import save_training_stats
            save_training_stats(
                self.train_loss_history, self.train_acc_history,
                self.test_loss_history, self.test_acc_history,
                self.best_acc, self.best_epoch, total_time, self.config
            )
            
            # Create plots
            create_training_plots(
                self.train_loss_history, self.train_acc_history,
                self.test_loss_history, self.test_acc_history,
                training_cfg['epochs_per_testing'],
                self.config['paths']['log_dir'],
                self.config['logging']['plot_format']
            )


def run_training(config: Dict[str, Any]):
    """Run optimized training with given configuration"""
    trainer = OptimizedTPUTrainer(config)
    trainer.train()