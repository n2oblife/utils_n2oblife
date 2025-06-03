import numpy as np
import logging
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import pickle
import json


class EarlyStopper:
    """
    Advanced Early Stopping implementation to prevent overfitting during training.
    
    This class monitors validation metrics and stops training when the model
    starts to overfit, with additional features for robust training control.
    """
    
    def __init__(self, 
                 patience: int = 7,
                 min_delta: float = 0.0,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 baseline: Optional[float] = None,
                 restore_best_weights: bool = True,
                 min_epochs: int = 0,
                 max_epochs: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize the Early Stopper.
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement
            monitor: Metric to monitor ('val_loss', 'val_accuracy', etc.)
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            baseline: Baseline value for the monitored metric. Training stops if not reached
            restore_best_weights: Whether to restore model weights from the best epoch
            min_epochs: Minimum number of epochs before early stopping can occur
            max_epochs: Maximum number of epochs to train
            verbose: Whether to print early stopping messages
        """
        # Validation
        if patience <= 0:
            raise ValueError("Patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        if min_epochs < 0:
            raise ValueError("min_epochs must be non-negative")
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        
        # Core parameters
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.verbose = verbose
        
        # State tracking
        self.wait = 0  # Number of epochs without improvement
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_weights = None
        self.history = []  # Track metric history
        
        # Initialize best value based on mode
        if mode == 'min':
            self.best_value = np.inf
            self.monitor_op = np.less
        else:
            self.best_value = -np.inf
            self.monitor_op = np.greater
        
        # Plateau detection
        self.plateau_patience = patience * 2  # Detect longer plateaus
        self.plateau_counter = 0
        self.plateau_threshold = min_delta * 0.1  # Smaller threshold for plateau
        
        # Additional stopping criteria
        self.divergence_threshold = None  # Set dynamically
        self.improvement_threshold = 0.01  # Minimum relative improvement
        
        # Statistics
        self.total_epochs = 0
        self.improvements = 0
        self.plateau_epochs = 0
    
    def __call__(self, 
                 current_value: float, 
                 epoch: int, 
                 model_weights: Optional[Any] = None) -> bool:
        """
        Check if training should stop based on current metric value.
        
        Args:
            current_value: Current value of the monitored metric
            epoch: Current epoch number
            model_weights: Current model weights (for restoration)
            
        Returns:
            True if training should stop, False otherwise
        """
        self.total_epochs = epoch + 1
        self.history.append(current_value)
        
        # Check minimum epochs requirement
        if epoch < self.min_epochs:
            if self.verbose and epoch == 0:
                print(f"EarlyStopper: Monitoring {self.monitor} with patience {self.patience}")
            return False
        
        # Check maximum epochs
        if self.max_epochs is not None and epoch >= self.max_epochs:
            if self.verbose:
                print(f"EarlyStopper: Maximum epochs ({self.max_epochs}) reached")
            self.stopped_epoch = epoch
            return True
        
        # Check baseline requirement
        if self.baseline is not None:
            if ((self.mode == 'min' and current_value > self.baseline) or
                (self.mode == 'max' and current_value < self.baseline)):
                if self.verbose:
                    print(f"EarlyStopper: Baseline {self.baseline} not met (current: {current_value:.6f})")
                return False
        
        # Check for improvement
        improved = self._check_improvement(current_value)
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            self.improvements += 1
            self.plateau_counter = 0
            
            # Store best weights if provided
            if model_weights is not None and self.restore_best_weights:
                self.best_weights = self._copy_weights(model_weights)
            
            if self.verbose:
                print(f"EarlyStopper: Improvement detected at epoch {epoch} "
                      f"({self.monitor}: {current_value:.6f})")
        else:
            self.wait += 1
            self._update_plateau_detection(current_value)
            
            if self.verbose and self.wait % 5 == 0:  # Print every 5 epochs without improvement
                print(f"EarlyStopper: No improvement for {self.wait} epochs "
                      f"(best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch})")
        
        # Check various stopping criteria
        should_stop = self._should_stop(current_value, epoch)
        
        if should_stop:
            self.stopped_epoch = epoch
            if self.verbose:
                self._print_stopping_reason(current_value, epoch)
        
        return should_stop
    
    def _check_improvement(self, current_value: float) -> bool:
        """Check if current value represents an improvement."""
        if self.monitor_op(current_value, self.best_value - self.min_delta):
            # Additional check for relative improvement on very small values
            if abs(self.best_value) > 1e-6:
                relative_improvement = abs((self.best_value - current_value) / self.best_value)
                return relative_improvement >= self.improvement_threshold
            return True
        return False
    
    def _update_plateau_detection(self, current_value: float):
        """Update plateau detection counters."""
        if len(self.history) >= 2:
            recent_change = abs(current_value - self.history[-2])
            if recent_change <= self.plateau_threshold:
                self.plateau_counter += 1
                self.plateau_epochs += 1
            else:
                self.plateau_counter = max(0, self.plateau_counter - 1)
    
    def _should_stop(self, current_value: float, epoch: int) -> bool:
        """Determine if training should stop based on various criteria."""
        # Standard patience-based stopping
        if self.wait >= self.patience:
            return True
        
        # Plateau detection
        if self.plateau_counter >= self.plateau_patience:
            if self.verbose:
                print(f"EarlyStopper: Training plateau detected for {self.plateau_counter} epochs")
            return True
        
        # Divergence detection (if loss is exploding)
        if self._check_divergence(current_value):
            return True
        
        # No significant improvement over extended period
        if self._check_stagnation():
            return True
        
        return False
    
    def _check_divergence(self, current_value: float) -> bool:
        """Check if the metric is diverging (getting much worse)."""
        if len(self.history) < 5:
            return False
        
        # Set divergence threshold dynamically
        if self.divergence_threshold is None:
            initial_values = self.history[:min(10, len(self.history))]
            self.divergence_threshold = np.std(initial_values) * 5 + abs(np.mean(initial_values))
        
        # Check if current value is much worse than best
        if self.mode == 'min':
            return current_value > (self.best_value + self.divergence_threshold)
        else:
            return current_value < (self.best_value - self.divergence_threshold)
    
    def _check_stagnation(self) -> bool:
        """Check for long-term stagnation."""
        if len(self.history) < self.patience * 3:
            return False
        
        # Check if there's been no significant improvement in a long time
        recent_window = self.history[-self.patience * 2:]
        if self.mode == 'min':
            recent_best = min(recent_window)
            return recent_best >= self.best_value * 0.995  # 0.5% tolerance
        else:
            recent_best = max(recent_window)
            return recent_best <= self.best_value * 1.005  # 0.5% tolerance
    
    def _copy_weights(self, weights: Any) -> Any:
        """Create a deep copy of model weights."""
        # This is a generic implementation - you might need to adapt based on your framework
        try:
            import copy
            return copy.deepcopy(weights)
        except:
            # Fallback for numpy arrays
            if hasattr(weights, 'copy'):
                return weights.copy()
            return weights
    
    def _print_stopping_reason(self, current_value: float, epoch: int):
        """Print detailed information about why training stopped."""
        print(f"\nEarlyStopper: Training stopped at epoch {epoch}")
        print(f"Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch}")
        print(f"Current {self.monitor}: {current_value:.6f}")
        print(f"Epochs without improvement: {self.wait}/{self.patience}")
        print(f"Total improvements: {self.improvements}")
        print(f"Plateau epochs: {self.plateau_epochs}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'stopped_epoch': self.stopped_epoch,
            'best_epoch': self.best_epoch,
            'best_value': self.best_value,
            'total_epochs': self.total_epochs,
            'improvements': self.improvements,
            'plateau_epochs': self.plateau_epochs,
            'final_patience_wait': self.wait,
            'history': self.history.copy()
        }
    
    def reset(self):
        """Reset the early stopper state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_weights = None
        self.history = []
        self.plateau_counter = 0
        self.total_epochs = 0
        self.improvements = 0
        self.plateau_epochs = 0
        self.divergence_threshold = None
        
        if self.mode == 'min':
            self.best_value = np.inf
        else:
            self.best_value = -np.inf
    
    def save_state(self, filepath: Union[str, Path]):
        """Save the current state to a file."""
        state = {
            'config': {
                'patience': self.patience,
                'min_delta': self.min_delta,
                'monitor': self.monitor,
                'mode': self.mode,
                'baseline': self.baseline,
                'min_epochs': self.min_epochs,
                'max_epochs': self.max_epochs
            },
            'state': self.get_statistics()
        }
        
        filepath = Path(filepath)
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
    
    def load_state(self, filepath: Union[str, Path]):
        """Load state from a file."""
        filepath = Path(filepath)
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                state = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
        
        # Restore configuration
        config = state['config']
        for key, value in config.items():
            setattr(self, key, value)
        
        # Restore state
        saved_state = state['state']
        self.stopped_epoch = saved_state['stopped_epoch']
        self.best_epoch = saved_state['best_epoch']
        self.best_value = saved_state['best_value']
        self.total_epochs = saved_state['total_epochs']
        self.improvements = saved_state['improvements']
        self.plateau_epochs = saved_state['plateau_epochs']
        self.wait = saved_state['final_patience_wait']
        self.history = saved_state['history']


# Specialized early stoppers for common use cases
class LossEarlyStopper(EarlyStopper):
    """Early stopper specifically for loss metrics."""
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, **kwargs):
        super().__init__(patience=patience, min_delta=min_delta, 
                        monitor='val_loss', mode='min', **kwargs)


class AccuracyEarlyStopper(EarlyStopper):
    """Early stopper specifically for accuracy metrics."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-3, **kwargs):
        super().__init__(patience=patience, min_delta=min_delta,
                        monitor='val_accuracy', mode='max', **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic usage
    early_stopper = EarlyStopper(patience=5, min_delta=0.001, verbose=True)
    
    # Simulate training with decreasing then increasing loss
    losses = [1.0, 0.8, 0.6, 0.4, 0.35, 0.34, 0.345, 0.35, 0.36, 0.37, 0.38]
    
    print("=== Testing Basic Early Stopping ===")
    for epoch, loss in enumerate(losses):
        should_stop = early_stopper(loss, epoch)
        if should_stop:
            print(f"Training stopped at epoch {epoch}")
            break
    
    print("\nStatistics:", early_stopper.get_statistics())
    
    # Example 2: Advanced usage with custom parameters
    print("\n=== Testing Advanced Early Stopping ===")
    advanced_stopper = EarlyStopper(
        patience=3,
        min_delta=0.01,
        monitor='val_accuracy',
        mode='max',
        baseline=0.8,
        min_epochs=5,
        verbose=True
    )
    
    # Simulate accuracy improvements
    accuracies = [0.5, 0.6, 0.7, 0.75, 0.82, 0.85, 0.86, 0.855, 0.854, 0.853]
    
    for epoch, acc in enumerate(accuracies):
        should_stop = advanced_stopper(acc, epoch)
        if should_stop:
            print(f"Training stopped at epoch {epoch}")
            break
    
    print("\nAdvanced Statistics:", advanced_stopper.get_statistics())