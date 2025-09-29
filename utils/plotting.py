"""
Plotting and visualization utilities
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def create_training_plots(train_loss_history, train_acc_history, 
                         test_loss_history, test_acc_history, 
                         epochs_per_testing, log_dir, 
                         plot_format='png'):
    """
    Create and save training history plots
    
    Args:
        train_loss_history: List of training losses
        train_acc_history: List of training accuracies
        test_loss_history: List of test losses
        test_acc_history: List of test accuracies
        epochs_per_testing: Interval between test evaluations
        log_dir: Directory to save plots
        plot_format: Format to save plots ('png', 'pdf', etc.)
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))

    # --- Plot Loss ---
    plt.subplot(1, 2, 1)
    # Plot training loss for every epoch
    plt.plot(np.arange(len(train_loss_history)), train_loss_history, 
             label='Train Loss', linewidth=2)

    # Plot test loss at testing intervals
    epochs = np.arange(len(test_loss_history))
    test_epochs = epochs * epochs_per_testing
    plt.plot(test_epochs, test_loss_history, 'o--', 
             label='Test Loss', 
             markevery=np.isfinite(test_loss_history), 
             markersize=5, linewidth=2)

    plt.title('Loss History Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot Accuracy ---
    plt.subplot(1, 2, 2)
    # Plot training accuracy for every epoch
    plt.plot(np.arange(len(train_acc_history)), 
             [acc * 100 for acc in train_acc_history], 
             label='Train Accuracy', linewidth=2)

    # Plot test accuracy at testing intervals
    plt.plot(test_epochs, 
             [acc * 100 if not np.isnan(acc) else np.nan for acc in test_acc_history], 
             'o--', label='Test Accuracy', 
             markevery=np.isfinite(test_acc_history), 
             markersize=5, linewidth=2)

    plt.title('Accuracy History Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plots
    loss_path = os.path.join(log_dir, f'loss_history.{plot_format}')
    training_path = os.path.join(log_dir, f'training_history.{plot_format}')
    
    # Save individual plots
    plt.subplot(1, 2, 1)
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    
    # Save combined plot
    plt.savefig(training_path, dpi=300, bbox_inches='tight')
    
    print(f"Training plots saved to {training_path}")


def plot_learning_curve(train_values, test_values, epochs_per_testing,
                       title, ylabel, log_dir, filename):
    """
    Plot a single learning curve (loss or accuracy)
    
    Args:
        train_values: Training values (per epoch)
        test_values: Test values (per epochs_per_testing)
        epochs_per_testing: Interval between test evaluations
        title: Plot title
        ylabel: Y-axis label
        log_dir: Directory to save plot
        filename: Filename for saving
    """
    os.makedirs(log_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot training curve
    plt.plot(range(len(train_values)), train_values, 
             label=f'Train {ylabel}', linewidth=2)
    
    # Plot test curve
    test_epochs = [i * epochs_per_testing for i in range(len(test_values))]
    valid_test_values = [v for v in test_values if not np.isnan(v)]
    valid_test_epochs = [e for i, e in enumerate(test_epochs) if not np.isnan(test_values[i])]
    
    plt.plot(valid_test_epochs, valid_test_values, 'o--', 
             label=f'Test {ylabel}', markersize=5, linewidth=2)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filepath = os.path.join(log_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath