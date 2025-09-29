"""
PyTorch TPU Trainer for ResNet18 on CIFAR-10
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


class TPUTrainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TPU Trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device, self.is_master = setup_tpu_device()
        
        # Training history
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []
        
        # Training state
        self.best_acc = 0.0
        self.best_epoch = -1
        self.current_epoch = 0
        
        # Get paths
        self.best_ckpt_path, self.last_ckpt_path = get_checkpoint_paths(config)
        
    def setup_data_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create data transforms based on config"""
        transform_cfg = self.config['transforms']
        
        # Training transforms
        train_transforms = []
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
        
        train_transforms.append(transforms.ToTensor())
        
        # Add normalization
        norm_cfg = transform_cfg['train']['normalize']
        train_transforms.append(
            transforms.Normalize(norm_cfg['mean'], norm_cfg['std'])
        )
        
        # Test transforms
        test_transforms = [transforms.ToTensor()]
        norm_cfg = transform_cfg['test']['normalize']
        test_transforms.append(
            transforms.Normalize(norm_cfg['mean'], norm_cfg['std'])
        )
        
        return transforms.Compose(train_transforms), transforms.Compose(test_transforms)
    
    def setup_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Setup data loaders"""
        transform_train, transform_test = self.setup_data_transforms()
        
        data_cfg = self.config['data']
        training_cfg = self.config['training']
        
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
        
        # Data loaders
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=training_cfg['batch_size'],
            shuffle=True,
            num_workers=data_cfg['num_workers'],
            drop_last=True
        )
        
        test_batch_size = training_cfg['batch_size'] * data_cfg['test_batch_multiplier']
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            drop_last=True
        )
        
        return trainloader, testloader
    
    def setup_model_and_optimizer(self) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Setup model, criterion, optimizer and scheduler"""
        model_cfg = self.config['model']
        training_cfg = self.config['training']
        
        # Model
        model = resnet18(num_classes=model_cfg['num_classes']).to(self.device)
        
        # Criterion
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer
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
        """Train for one epoch"""
        model.train()
        total_train_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0
        
        train_device_loader = create_parallel_loader(trainloader, self.device)
        
        for inputs, targets in train_device_loader:
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            
            optimizer_step(optimizer)
            
            # Collect metrics locally
            total_train_loss += loss.item() * inputs.size(0)
            _, predicted = logits.max(1)
            train_correct_predictions += predicted.eq(targets).sum().item()
            train_total_samples += targets.size(0)
            
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
    
    def evaluate(self, model: nn.Module, criterion: nn.Module, 
                testloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Evaluate the model"""
        model.eval()
        total_test_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        test_device_loader = create_parallel_loader(testloader, self.device)
        
        with torch.no_grad():
            for inputs, targets in test_device_loader:
                logits = model(inputs)
                loss = criterion(logits, targets)
                
                total_test_loss += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                total_samples += targets.size(0)
                correct_predictions += predicted.eq(targets).sum().item()
        
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
        """Save model checkpoint"""
        if not self.is_master:
            return
        
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': self.best_acc
        }
        
        if is_best:
            save_checkpoint(state, self.best_ckpt_path)
        else:
            save_checkpoint(state, self.last_ckpt_path)
    
    def train(self):
        """Main training loop"""
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
            
            # Warmup logic
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
            
            self.current_epoch = epoch
        
        # Final logging and saving
        if self.is_master:
            # Save final checkpoint
            self.save_checkpoint(model, optimizer, scheduler, self.current_epoch + 1)
            
            # Log final results
            total_time = (datetime.now() - start_time).total_seconds()
            log_final_results(self.best_acc, self.best_epoch, total_time, self.is_master)
            
            # Create plots
            create_training_plots(
                self.train_loss_history, self.train_acc_history,
                self.test_loss_history, self.test_acc_history,
                training_cfg['epochs_per_testing'],
                self.config['paths']['log_dir'],
                self.config['logging']['plot_format']
            )


def run_training(config: Dict[str, Any]):
    """Run training with given configuration"""
    trainer = TPUTrainer(config)
    trainer.train()