import torch
import signal
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    """Complete training state container"""
    model_name: str
    current_epoch: int
    total_epochs: int
    current_step: int
    best_metric: float
    best_epoch: int
    learning_rate: float
    batch_size: int
    dataset_info: Dict[str, Any]
    training_config: Dict[str, Any]
    timestamp: float
    
    def to_dict(self):
        return asdict(self)

class TrainingInterruptHandler:
    """
    Comprehensive interrupt handler for training processes
    Handles Ctrl+C gracefully and saves all training state
    """
    
    def __init__(self, 
                 save_dir: str = "./checkpoints",
                 auto_save_interval: int = 10,  # Save every N epochs
                 prompt_on_interrupt: bool = True):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save_interval = auto_save_interval
        self.prompt_on_interrupt = prompt_on_interrupt
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision training
        self.evaluator = None
        self.training_state = None
        
        # Interrupt handling
        self.interrupted = False
        self.original_sigint_handler = None
        
        # Setup signal handler
        self.setup_interrupt_handler()
        
        logger.info(f"Interrupt handler initialized. Save directory: {self.save_dir}")
    
    def setup_interrupt_handler(self):
        """Setup signal handler for Ctrl+C"""
        self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle interrupt signal"""
        print("\n" + "="*60)
        print("🛑 TRAINING INTERRUPTED!")
        print("="*60)
        
        self.interrupted = True
        
        if self.prompt_on_interrupt:
            self._interactive_save()
        else:
            self._auto_save()
        
        # Restore original handler and re-raise
        signal.signal(signal.SIGINT, self.original_sigint_handler)
        sys.exit(0)
    
    def register_training_components(self,
                                   model: torch.nn.Module,
                                   optimizer: torch.optim.Optimizer,
                                   training_state: TrainingState,
                                   evaluator=None,
                                   scheduler=None,
                                   scaler=None):
        """Register all training components for saving"""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.evaluator = evaluator
        self.training_state = training_state
        
        logger.info("Training components registered with interrupt handler")
    
    def update_training_state(self, 
                            current_epoch: int,
                            current_step: int,
                            best_metric: float = None,
                            best_epoch: int = None,
                            **kwargs):
        """Update current training state"""
        if self.training_state:
            self.training_state.current_epoch = current_epoch
            self.training_state.current_step = current_step
            if best_metric is not None:
                self.training_state.best_metric = best_metric
            if best_epoch is not None:
                self.training_state.best_epoch = best_epoch
            
            # Update any additional fields
            for key, value in kwargs.items():
                if hasattr(self.training_state, key):
                    setattr(self.training_state, key, value)
    
    def _interactive_save(self):
        """Interactive saving with user prompts"""
        print(f"\n📊 Current Training Status:")
        print(f"   Model: {self.training_state.model_name}")
        print(f"   Epoch: {self.training_state.current_epoch}/{self.training_state.total_epochs}")
        print(f"   Step: {self.training_state.current_step}")
        print(f"   Best Metric: {self.training_state.best_metric:.4f} (Epoch {self.training_state.best_epoch})")
        
        print(f"\n💾 Save Options:")
        print("1. Save everything (model + optimizer + metrics + full state)")
        print("2. Save model only (lightest)")
        print("3. Save model + metrics")
        print("4. Don't save (exit without saving)")
        print("5. Continue training")
        
        while True:
            try:
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == '1':
                    self._save_complete_checkpoint()
                    break
                elif choice == '2':
                    self._save_model_only()
                    break
                elif choice == '3':
                    self._save_model_and_metrics()
                    break
                elif choice == '4':
                    print("❌ Exiting without saving")
                    break
                elif choice == '5':
                    print("▶️ Continuing training...")
                    self.interrupted = False
                    return
                else:
                    print("❗ Invalid choice. Please select 1-5.")
                    
            except KeyboardInterrupt:
                print("\n❌ Second interrupt detected. Force exiting without save.")
                break
            except Exception as e:
                print(f"❗ Error reading input: {e}")
                break
    
    def _auto_save(self):
        """Automatic saving without prompts"""
        print("💾 Auto-saving complete checkpoint...")
        self._save_complete_checkpoint()
    
    def _save_complete_checkpoint(self):
        """Save complete training checkpoint"""
        try:
            timestamp = int(time.time())
            checkpoint_name = f"{self.training_state.model_name}_interrupt_epoch_{self.training_state.current_epoch}_{timestamp}"
            
            # Prepare checkpoint data
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_state': self.training_state.to_dict(),
                'epoch': self.training_state.current_epoch,
                'step': self.training_state.current_step,
                'best_metric': self.training_state.best_metric,
                'best_epoch': self.training_state.best_epoch,
                'timestamp': timestamp,
                'pytorch_version': torch.__version__,
                'checkpoint_type': 'interrupt_complete'
            }
            
            # Add scheduler state if available
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # Add scaler state if available (mixed precision)
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            # Add evaluation metrics if available
            if self.evaluator is not None:
                try:
                    # Save training history
                    self.evaluator.tracker.save_history(f"{checkpoint_name}_training_history.json")
                    
                    # Get best metrics
                    best_metrics = {}
                    for metric in ['accuracy', 'loss', 'f1_score']:
                        best = self.evaluator.tracker.get_best_metrics(metric, mode='max' if metric != 'loss' else 'min')
                        if best:
                            best_metrics[f'best_{metric}'] = best
                    
                    checkpoint['best_metrics'] = best_metrics
                    checkpoint['metrics_history_file'] = f"{checkpoint_name}_training_history.json"
                    
                except Exception as e:
                    logger.warning(f"Could not save evaluation metrics: {e}")
            
            # Save checkpoint
            checkpoint_path = self.save_dir / f"{checkpoint_name}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Save human-readable summary
            self._save_checkpoint_summary(checkpoint_name, checkpoint)
            
            print(f"✅ Complete checkpoint saved:")
            print(f"   📁 {checkpoint_path}")
            print(f"   📊 Including: model, optimizer, scheduler, metrics, full state")
            
        except Exception as e:
            logger.error(f"Failed to save complete checkpoint: {e}")
            print(f"❌ Failed to save checkpoint: {e}")
    
    def _save_model_only(self):
        """Save only the model state"""
        try:
            timestamp = int(time.time())
            model_name = f"{self.training_state.model_name}_model_only_epoch_{self.training_state.current_epoch}_{timestamp}.pth"
            model_path = self.save_dir / model_name
            
            # Save only model state dict
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_name': self.training_state.model_name,
                'epoch': self.training_state.current_epoch,
                'timestamp': timestamp,
                'checkpoint_type': 'model_only'
            }, model_path)
            
            print(f"✅ Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            print(f"❌ Failed to save model: {e}")
    
    def _save_model_and_metrics(self):
        """Save model and metrics only"""
        try:
            timestamp = int(time.time())
            checkpoint_name = f"{self.training_state.model_name}_model_metrics_epoch_{self.training_state.current_epoch}_{timestamp}"
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'training_state': self.training_state.to_dict(),
                'epoch': self.training_state.current_epoch,
                'best_metric': self.training_state.best_metric,
                'best_epoch': self.training_state.best_epoch,
                'timestamp': timestamp,
                'checkpoint_type': 'model_and_metrics'
            }
            
            # Add metrics if evaluator is available
            if self.evaluator is not None:
                self.evaluator.tracker.save_history(f"{checkpoint_name}_metrics.json")
                checkpoint['metrics_file'] = f"{checkpoint_name}_metrics.json"
            
            checkpoint_path = self.save_dir / f"{checkpoint_name}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            print(f"✅ Model and metrics saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model and metrics: {e}")
            print(f"❌ Failed to save model and metrics: {e}")
    
    def _save_checkpoint_summary(self, checkpoint_name: str, checkpoint: Dict):
        """Save human-readable checkpoint summary"""
        try:
            summary = {
                'checkpoint_info': {
                    'name': checkpoint_name,
                    'type': checkpoint.get('checkpoint_type', 'unknown'),
                    'saved_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint['timestamp'])),
                    'pytorch_version': checkpoint.get('pytorch_version', 'unknown')
                },
                'training_progress': {
                    'current_epoch': checkpoint['epoch'],
                    'total_epochs': self.training_state.total_epochs,
                    'progress_percentage': (checkpoint['epoch'] / self.training_state.total_epochs) * 100,
                    'current_step': checkpoint['step']
                },
                'performance': {
                    'best_metric': checkpoint['best_metric'],
                    'best_epoch': checkpoint['best_epoch'],
                    'current_lr': self.training_state.learning_rate
                },
                'model_info': {
                    'name': self.training_state.model_name,
                    'parameters': sum(p.numel() for p in self.model.parameters()),
                    'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                }
            }
            
            summary_path = self.save_dir / f"{checkpoint_name}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save checkpoint summary: {e}")
    
    def auto_save_checkpoint(self, epoch: int):
        """Automatically save checkpoint at specified intervals"""
        if epoch % self.auto_save_interval == 0:
            try:
                checkpoint_name = f"{self.training_state.model_name}_auto_epoch_{epoch}"
                checkpoint_path = self.save_dir / f"{checkpoint_name}.pth"
                
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_state': self.training_state.to_dict(),
                    'epoch': epoch,
                    'timestamp': time.time(),
                    'checkpoint_type': 'auto_save'
                }
                
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Auto-saved checkpoint at epoch {epoch}: {checkpoint_path}")
                
            except Exception as e:
                logger.error(f"Auto-save failed at epoch {epoch}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer = None,
                       scheduler = None) -> Dict:
        """Load checkpoint and restore training state"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if available
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore training state
            if 'training_state' in checkpoint:
                training_state_dict = checkpoint['training_state']
                self.training_state = TrainingState(**training_state_dict)
            
            logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            logger.info(f"Resuming from epoch {checkpoint.get('epoch', 'unknown')}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def cleanup(self):
        """Cleanup and restore original signal handler"""
        if self.original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
        
        logger.info("Interrupt handler cleaned up")

# Context manager for easy usage
class TrainingSession:
    """Context manager for training with interrupt handling"""
    
    def __init__(self, handler: TrainingInterruptHandler):
        self.handler = handler
    
    def __enter__(self):
        return self.handler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handler.cleanup()
        return False  # Don't suppress exceptions

# Example usage
if __name__ == "__main__":
    # Example of how to use the interrupt handler
    
    # Create training state
    training_state = TrainingState(
        model_name="resnet50_classifier",
        current_epoch=0,
        total_epochs=100,
        current_step=0,
        best_metric=0.0,
        best_epoch=0,
        learning_rate=0.001,
        batch_size=32,
        dataset_info={"num_samples": 50000, "num_classes": 10},
        training_config={"optimizer": "Adam", "scheduler": "CosineAnnealing"},
        timestamp=time.time()
    )
    
    # Create interrupt handler
    handler = TrainingInterruptHandler(
        save_dir="./checkpoints",
        auto_save_interval=5,
        prompt_on_interrupt=True
    )
    
    print("Interrupt handler example created!")
    print("In your training loop, you would:")
    print("1. Register components with handler.register_training_components(...)")
    print("2. Update state with handler.update_training_state(...)")
    print("3. Use handler.auto_save_checkpoint(epoch) for regular saves")
    print("4. Ctrl+C will trigger interactive save options")