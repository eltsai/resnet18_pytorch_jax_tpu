"""
JAX/Flax TPU Trainer for ResNet18 on CIFAR-10
"""
import jax
import jax.numpy as jnp
import jax.lax
import optax
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List
from functools import partial

from models.jax.resnet18 import resnet18
from utils.jax_utils_optimized import (
    TrainState, setup_jax_tpu, save_checkpoint, aggregate_metrics_across_devices, get_checkpoint_paths, print_tpu_info
)
from utils.jax_streaming_loader import create_streaming_data_loader
from utils.logging_utils import (
    log_training_info, log_epoch_info, log_metrics,
    log_best_checkpoint, log_final_results, save_training_stats
)
from utils.plotting import create_training_plots


class JAXTPUTrainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize JAX TPU Trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.num_devices, self.is_tpu = setup_jax_tpu()
        
        # Training history
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []
        
        # Training state
        self.best_acc = 0.0
        self.best_epoch = -1
        self.current_epoch = 0
        
        # Data loader - will be set up once in train()
        self.data_loader = None
        
        # Get paths
        self.best_ckpt_path, self.last_ckpt_path = get_checkpoint_paths(config)
        
        # Pre-compile pmap functions to avoid jit-of-pmap warnings
        # Note: These will be initialized after train_step and eval_step are defined
        self.pmapped_train_step = None
        self.pmapped_eval_step = None
    
    def _initialize_pmapped_functions(self):
        """Initialize pmapped functions after methods are defined"""
        if self.num_devices > 1:
            self.pmapped_train_step = jax.pmap(self.train_step, static_broadcasted_argnums=(2,))
            self.pmapped_eval_step = jax.pmap(self.eval_step, static_broadcasted_argnums=(2,))
        
    def create_learning_rate_schedule(self, total_steps: int, steps_per_epoch: int):
        """Create learning rate schedule with warmup and cosine decay"""
        training_cfg = self.config['training']
        warmup_epochs = training_cfg['warmup_epochs']
        base_lr = training_cfg['base_lr']
        eta_min = training_cfg['scheduler_eta_min']
        
        warmup_steps = warmup_epochs * steps_per_epoch
        
        warmup_schedule = optax.linear_schedule(
            init_value=1e-7,
            end_value=base_lr,
            transition_steps=warmup_steps
        )
        
        cosine_steps = total_steps - warmup_steps
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=cosine_steps,
            alpha=eta_min / base_lr
        )
        
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[warmup_steps]
        )
        return schedule
    
    def create_optimizer(self, learning_rate_fn):
        """Create optimizer with SGD + momentum + weight decay"""
        training_cfg = self.config['training']
        
        optimizer = optax.chain(
            optax.add_decayed_weights(training_cfg['weight_decay']),
            optax.sgd(
                learning_rate=learning_rate_fn,
                momentum=training_cfg['momentum'],
                nesterov=training_cfg['nesterov']
            )
        )
        return optimizer
    
    def setup_model_and_optimizer(self) -> Tuple[Any, TrainState]:
        """Setup model, optimizer and training state"""
        model_cfg = self.config['model']
        training_cfg = self.config['training']
        
        # Create model
        model = resnet18(num_classes=model_cfg['num_classes'])
        
        # Initialize model
        key = jax.random.PRNGKey(self.config['logging']['seed'])
        per_device_batch_size = training_cfg['per_device_batch_size']
        
        dummy_input = jnp.ones((per_device_batch_size, 32, 32, 3), dtype=jnp.float32)
        initial_variables = model.init(key, dummy_input, train=True)
        
        # Calculate training steps and create optimizer
        train_samples = 50000  # CIFAR-10 training set size
        global_batch_size = per_device_batch_size * self.num_devices
        steps_per_epoch = train_samples // global_batch_size
        total_steps = training_cfg['total_epochs'] * steps_per_epoch
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        
        lr_schedule = self.create_learning_rate_schedule(total_steps, steps_per_epoch)
        optimizer = self.create_optimizer(lr_schedule)
        
        # Create training state
        state = TrainState.create(
            apply_fn=model.apply,
            params=initial_variables['params'],
            tx=optimizer,
            batch_stats=initial_variables['batch_stats']
        )
        
        # Replicate state across devices if multi-device
        if self.num_devices > 1:
            state = jax.device_put_replicated(state, jax.devices())
            
        return model, state, lr_schedule
    
    @partial(jax.jit, static_argnames=['self', 'model'])
    def cross_entropy_loss(self, logits, targets, model):
        """Cross entropy loss function"""
        one_hot_targets = jax.nn.one_hot(targets, num_classes=model.num_classes)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_targets)
        return jnp.mean(loss)
    
    @partial(jax.jit, static_argnames=['self', 'model'])
    def train_step(self, state: TrainState, batch, model):
        """JIT-compiled training step"""
        inputs, targets = batch
        
        def loss_fn(params):
            logits, new_model_state = model.apply(
                {'params': params, 'batch_stats': state.batch_stats},
                inputs,
                train=True,
                mutable=['batch_stats']
            )
            loss = self.cross_entropy_loss(logits, targets, model)
            return loss, (logits, new_model_state)

        (loss, (logits, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
        
        accuracy = jnp.mean(jnp.argmax(logits, -1) == targets)
        
        return state, loss, accuracy
    
    @partial(jax.jit, static_argnames=['self', 'model'])
    def eval_step(self, state: TrainState, batch, model):
        """JIT-compiled evaluation step"""
        inputs, targets = batch
        
        # For evaluation: allow batch_stats to be mutable to avoid the error
        # But we won't use the updated batch_stats (they'll be discarded)
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        logits, updated_vars = model.apply(
            variables,
            inputs,
            train=False,  # Use running averages for BatchNorm
            mutable=['batch_stats']  # Allow batch_stats updates (but we ignore them)
        )
        
        one_hot_targets = jax.nn.one_hot(targets, num_classes=model.num_classes)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_targets)
        loss = jnp.mean(loss)
        
        accuracy = jnp.mean(jnp.argmax(logits, -1) == targets)
        
        return loss, accuracy
    
    @partial(jax.jit, static_argnames=['self', 'model'])
    def train_epoch_scan_step(self, carry_state, batch, model):
        """Single training step for JAX scan - JIT compiled"""
        state = carry_state
        inputs, targets = batch
        
        def loss_fn(params):
            logits, new_model_state = model.apply(
                {'params': params, 'batch_stats': state.batch_stats},
                inputs,
                train=True,
                mutable=['batch_stats']
            )
            one_hot_targets = jax.nn.one_hot(targets, num_classes=model.num_classes)
            loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_targets)
            loss = jnp.mean(loss)
            return loss, (logits, new_model_state)

        (loss, (logits, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
        accuracy = jnp.mean(jnp.argmax(logits, -1) == targets)
        
        return new_state, (loss, accuracy)
    

    
    def single_batch_step(self, state, batch, model):
        """Single batch step - uses pre-compiled pmap to avoid jit-of-pmap warning"""
        inputs, targets = batch
        
        if self.num_devices > 1:
            # Use pre-compiled pmap function
            new_state, loss, acc = self.pmapped_train_step(state, (inputs, targets), model)
            # Average across devices
            return new_state, jnp.mean(loss), jnp.mean(acc)
        else:
            return self.train_step(state, (inputs, targets), model)



    def train_epoch(self, model, state, lr_schedule, epoch: int) -> Tuple[TrainState, float, float]:
        """Train for one epoch - simplified JAX approach"""
        print(f"\n=== OPTIMIZED JAX TRAINING EPOCH {epoch + 1} ===")
        epoch_start_time = time.time()
        
        # For now, let's use individual JIT-compiled steps instead of full scan
        # This avoids the pmap/scan dimension mismatch while still getting JIT benefits
        
        total_train_loss = 0.0
        total_train_acc = 0.0
        train_batches = 0
        
        batch_start_time = time.time()
        current_state = state
        
        print("Running JIT-compiled training steps...")
        
        # Process batches with JIT compilation
        for batch_data, batch_labels in self.data_loader.get_train_batches(epoch, self.config['logging']['seed']):
            
            # Use JIT-compiled single batch step
            current_state, loss, acc = self.single_batch_step(current_state, (batch_data, batch_labels), model)
            
            total_train_loss += loss
            total_train_acc += acc
            train_batches += 1
            
            # Print progress every 10 batches with timing
            if train_batches % 10 == 0:
                batch_time = time.time() - batch_start_time
                avg_batch_time = batch_time / 10
                steps_per_epoch = self.data_loader.steps_per_epoch
                print(f"  Batch {train_batches}/{steps_per_epoch}: loss={float(loss):.4f}, acc={float(acc):.4f}, "
                      f"avg_batch_time={avg_batch_time:.3f}s")
                batch_start_time = time.time()
        
        total_time = time.time() - epoch_start_time
        
        avg_train_loss = total_train_loss / train_batches
        avg_train_acc = total_train_acc / train_batches
        
        print(f"Total epoch time: {total_time:.2f}s")
        print(f"Avg batch time: {total_time / train_batches:.3f}s")
        print(f"Epoch {epoch + 1} completed - Loss: {float(avg_train_loss):.4f}, Acc: {float(avg_train_acc):.4f}")
        
        # Get learning rate
        if self.num_devices == 1:
            current_lr = lr_schedule(current_state.step)
        else:
            current_lr = lr_schedule(current_state.step[0])  # All devices have same step count
        
        return current_state, float(avg_train_loss), float(avg_train_acc), float(current_lr)
    
    def evaluate(self, model, state) -> Tuple[float, float]:
        """Evaluate the model - optimized version"""
        total_test_loss = 0.0
        total_test_acc = 0.0
        test_batches = 0
        
        for batch_data, batch_labels in self.data_loader.get_test_batches():
            
            # Data is already properly shaped for multi-device by data loader
            if self.num_devices > 1:
                loss, acc = self.pmapped_eval_step(state, (batch_data, batch_labels), model)
                loss = jnp.mean(loss)
                acc = jnp.mean(acc)
            else:
                loss, acc = self.eval_step(state, (batch_data, batch_labels), model)
            
            total_test_loss += loss
            total_test_acc += acc
            test_batches += 1
        
        avg_test_loss = total_test_loss / test_batches
        avg_test_acc = total_test_acc / test_batches
        
        return float(avg_test_loss), float(avg_test_acc)
    
    def save_checkpoint(self, state: TrainState, is_best: bool = False):
        """Save model checkpoint"""
        if is_best:
            save_checkpoint(state, self.best_ckpt_path)
        else:
            save_checkpoint(state, self.last_ckpt_path)
    
    def train(self):
        """Main training loop"""
        # Setup
        print_tpu_info(self.num_devices, self.is_tpu)
        log_training_info(self.config, True)  # Always log for JAX (single process)
        
        # Initialize pmapped functions now that methods are defined
        self._initialize_pmapped_functions()
        
        # Setup model and data
        model, state, lr_schedule = self.setup_model_and_optimizer()
        self.data_loader = create_streaming_data_loader(self.config)
        
        training_cfg = self.config['training']
        start_time = datetime.now()
        
        print(f"\nStarting training for {training_cfg['total_epochs']} epochs...")
        
        # Training loop
        for epoch in range(training_cfg['total_epochs']):
            epoch_start_time = time.time()
            
            # Log epoch info
            state, avg_train_loss, train_acc, current_lr = self.train_epoch(
                model, state, lr_schedule, epoch
            )
            
            log_epoch_info(epoch, training_cfg['total_epochs'], current_lr, True, epoch == 0)
            
            self.train_loss_history.append(avg_train_loss)
            self.train_acc_history.append(train_acc)
            
            # Evaluation and checkpointing
            should_evaluate = ((epoch + 1) % training_cfg['epochs_per_testing'] == 0 or 
                             (epoch + 1) == training_cfg['total_epochs'])
            
            if should_evaluate:
                avg_test_loss, test_acc = self.evaluate(model, state)
                
                self.test_loss_history.append(avg_test_loss)
                self.test_acc_history.append(test_acc)
                
                # Check for best model
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    self.best_epoch = epoch + 1
                    
                    # Update state with best metrics
                    if self.num_devices == 1:
                        state = state.replace(best_acc=test_acc, best_epoch=epoch + 1)
                    else:
                        # For pmap, we need to update the replicated state correctly
                        # The state is already replicated across devices, so we update it on one device
                        # and then replicate it again
                        first_device_state = jax.tree_util.tree_map(lambda x: x[0], state)
                        updated_state = first_device_state.replace(best_acc=test_acc, best_epoch=epoch + 1)
                        state = jax.device_put_replicated(updated_state, jax.devices())
                    
                    self.save_checkpoint(state, is_best=True)
                    log_best_checkpoint(self.best_epoch, self.best_acc, True)
                
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                log_metrics(epoch, avg_train_loss, train_acc, 
                           avg_test_loss, test_acc, epoch_time, True)
            else:
                self.test_loss_history.append(np.nan)
                self.test_acc_history.append(np.nan)
                
                epoch_time = time.time() - epoch_start_time
                log_metrics(epoch, avg_train_loss, train_acc, 
                           epoch_time=epoch_time, is_master=True)
            
            self.current_epoch = epoch
        
        # Final logging and saving
        # Save final checkpoint
        self.save_checkpoint(state)
        
        # Log final results
        total_time = (datetime.now() - start_time).total_seconds()
        log_final_results(self.best_acc, self.best_epoch, total_time, True)
        
        # Save training statistics
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
    """Run training with given configuration"""
    trainer = JAXTPUTrainer(config)
    trainer.train()