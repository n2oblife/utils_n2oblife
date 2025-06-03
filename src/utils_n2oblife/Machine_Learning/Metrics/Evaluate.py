import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    iou: float = 0.0
    dice_coefficient: float = 0.0
    loss: float = 0.0
    inference_time: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        base_dict = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'iou': self.iou,
            'dice_coefficient': self.dice_coefficient,
            'loss': self.loss,
            'inference_time': self.inference_time
        }
        base_dict.update(self.custom_metrics)
        return base_dict

@dataclass
class ModelReference:
    """Lightweight model reference instead of storing the full model"""
    model_name: str = None
    model_path: str = None
    model_type: str = None
    model_size: str = None
    architecture: str = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def load_model(self, device: str = 'cpu'):
        """Load model from path when needed"""
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} not found")
        
        try:
            if self.model_path.endswith(('.pt', '.pth', '.pwf', '.ptc')):
                checkpoint = torch.load(self.model_path, map_location=device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        return checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        return checkpoint['model']
                    else:
                        return checkpoint
                else:
                    return checkpoint
                    
            elif self.model_path.endswith('.engine'):
                # TensorRT engine loading
                raise NotImplementedError("TensorRT engine loading not implemented")
                
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

class TrainingTracker:
    """Track metrics during training"""
    
    def __init__(self, save_dir: str = "./evaluation_logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.training_history = defaultdict(list)
        self.validation_history = defaultdict(list)
        self.current_epoch = 0
        
    def log_training_step(self, metrics: Dict[str, float], step: int):
        """Log metrics for a training step"""
        for key, value in metrics.items():
            self.training_history[f"train_{key}"].append({
                'step': step,
                'epoch': self.current_epoch,
                'value': value,
                'timestamp': time.time()
            })
    
    def log_validation_step(self, metrics: Dict[str, float], step: int):
        """Log metrics for a validation step"""
        for key, value in metrics.items():
            self.validation_history[f"val_{key}"].append({
                'step': step,
                'epoch': self.current_epoch,
                'value': value,
                'timestamp': time.time()
            })
    
    def next_epoch(self):
        """Move to next epoch"""
        self.current_epoch += 1
    
    def save_history(self, filename: str = None):
        """Save training history to file"""
        if filename is None:
            filename = f"training_history_epoch_{self.current_epoch}.json"
        
        filepath = self.save_dir / filename
        history = {
            'training': dict(self.training_history),
            'validation': dict(self.validation_history),
            'current_epoch': self.current_epoch
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to {filepath}")
    
    def get_best_metrics(self, metric_name: str, mode: str = 'max') -> Dict:
        """Get best metric value and corresponding epoch"""
        history_key = f"val_{metric_name}"
        if history_key not in self.validation_history:
            return None
        
        values = [entry['value'] for entry in self.validation_history[history_key]]
        if not values:
            return None
        
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        best_entry = self.validation_history[history_key][best_idx]
        return {
            'value': best_entry['value'],
            'epoch': best_entry['epoch'],
            'step': best_entry['step']
        }

class BaseEvaluator(ABC):
    """Abstract base class for model evaluation"""
    
    def __init__(self, 
                 model_ref: ModelReference,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 32):
        self.model_ref = model_ref
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.metrics_history = []
        self.tracker = TrainingTracker()
        
    def load_model(self):
        """Load model when needed"""
        if self.model is None:
            self.model = self.model_ref.load_model(self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
        return self.model
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @abstractmethod
    def evaluate_batch(self, batch_data, batch_labels) -> EvaluationMetrics:
        """Evaluate a single batch - to be implemented by subclasses"""
        pass
    
    def evaluate_dataset(self, dataloader, 
                        save_results: bool = True,
                        progress_desc: str = None) -> EvaluationMetrics:
        """Evaluate entire dataset"""
        self.load_model()
        
        if progress_desc is None:
            progress_desc = f"Evaluating {self.model_ref.model_name}"
        
        all_metrics = []
        total_samples = 0
        
        with torch.no_grad():
            progress = tqdm(dataloader, desc=progress_desc, ncols=100)
            
            for batch_data, batch_labels in progress:
                # Move data to device
                if isinstance(batch_data, torch.Tensor):
                    batch_data = batch_data.to(self.device)
                if isinstance(batch_labels, torch.Tensor):
                    batch_labels = batch_labels.to(self.device)
                
                # Evaluate batch
                batch_metrics = self.evaluate_batch(batch_data, batch_labels)
                all_metrics.append(batch_metrics)
                total_samples += len(batch_labels)
                
                # Update progress bar
                current_acc = batch_metrics.accuracy
                progress.set_postfix({'Accuracy': f'{current_acc:.3f}'})
        
        # Aggregate metrics
        final_metrics = self._aggregate_metrics(all_metrics, total_samples)
        
        if save_results:
            self._save_evaluation_results(final_metrics)
        
        self.metrics_history.append(final_metrics)
        return final_metrics
    
    def _aggregate_metrics(self, metrics_list: List[EvaluationMetrics], 
                          total_samples: int) -> EvaluationMetrics:
        """Aggregate metrics from all batches"""
        if not metrics_list:
            return EvaluationMetrics()
        
        # Weighted average based on batch sizes
        aggregated = EvaluationMetrics()
        
        # Simple average for now - can be improved with proper weighting
        aggregated.accuracy = np.mean([m.accuracy for m in metrics_list])
        aggregated.precision = np.mean([m.precision for m in metrics_list])
        aggregated.recall = np.mean([m.recall for m in metrics_list])
        aggregated.f1_score = np.mean([m.f1_score for m in metrics_list])
        aggregated.iou = np.mean([m.iou for m in metrics_list])
        aggregated.dice_coefficient = np.mean([m.dice_coefficient for m in metrics_list])
        aggregated.loss = np.mean([m.loss for m in metrics_list])
        aggregated.inference_time = np.sum([m.inference_time for m in metrics_list])
        
        # Aggregate custom metrics
        all_custom_keys = set()
        for m in metrics_list:
            all_custom_keys.update(m.custom_metrics.keys())
        
        for key in all_custom_keys:
            values = [m.custom_metrics.get(key, 0) for m in metrics_list]
            aggregated.custom_metrics[key] = np.mean(values)
        
        return aggregated
    
    def _save_evaluation_results(self, metrics: EvaluationMetrics):
        """Save evaluation results"""
        results = {
            'model_info': {
                'name': self.model_ref.model_name,
                'path': self.model_ref.model_path,
                'type': self.model_ref.model_type,
                'architecture': self.model_ref.architecture
            },
            'metrics': metrics.to_dict(),
            'timestamp': time.time(),
            'device': self.device
        }
        
        filename = f"evaluation_{self.model_ref.model_name}_{int(time.time())}.json"
        filepath = self.tracker.save_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def track_training_progress(self, train_metrics: Dict[str, float], 
                              val_metrics: Dict[str, float], step: int):
        """Track metrics during training"""
        self.tracker.log_training_step(train_metrics, step)
        self.tracker.log_validation_step(val_metrics, step)
    
    def end_epoch(self):
        """Called at the end of each training epoch"""
        self.tracker.next_epoch()
        self.tracker.save_history()


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks"""
    
    def __init__(self, model_ref: ModelReference, num_classes: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 32):
        super().__init__(model_ref, device, batch_size)
        self.num_classes = num_classes
    
    def evaluate_batch(self, batch_data, batch_labels) -> EvaluationMetrics:
        """Evaluate classification batch"""
        start_time = time.time()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(batch_data)
            
            # Calculate loss if criterion is available
            loss = 0.0
            if hasattr(self, 'criterion'):
                loss = self.criterion(outputs, batch_labels).item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
        inference_time = time.time() - start_time
        
        # Calculate metrics
        correct = (predicted == batch_labels).sum().item()
        total = batch_labels.size(0)
        accuracy = correct / total
        
        # Calculate precision, recall, F1 (simplified version)
        precision = recall = f1_score = accuracy  # Simplified for demo
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            loss=loss,
            inference_time=inference_time
        )


class SegmentationEvaluator(BaseEvaluator):
    """Evaluator for segmentation tasks"""
    
    def __init__(self, model_ref: ModelReference, num_classes: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 32):
        super().__init__(model_ref, device, batch_size)
        self.num_classes = num_classes
    
    def evaluate_batch(self, batch_data, batch_labels) -> EvaluationMetrics:
        """Evaluate segmentation batch"""
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(batch_data)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                outputs = outputs['out']  # DeepLab format
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
        inference_time = time.time() - start_time
        
        # Calculate IoU and Dice coefficient
        iou = self._calculate_iou(predictions, batch_labels)
        dice = self._calculate_dice(predictions, batch_labels)
        
        # Pixel accuracy
        correct_pixels = (predictions == batch_labels).sum().item()
        total_pixels = batch_labels.numel()
        accuracy = correct_pixels / total_pixels
        
        return EvaluationMetrics(
            accuracy=accuracy,
            iou=iou,
            dice_coefficient=dice,
            inference_time=inference_time
        )
    
    def _calculate_iou(self, predictions, targets):
        """Calculate Intersection over Union"""
        intersection = (predictions & targets).float().sum()
        union = (predictions | targets).float().sum()
        
        if union == 0:
            return 1.0  # If both are empty, IoU is 1
        
        return (intersection / union).item()
    
    def _calculate_dice(self, predictions, targets):
        """Calculate Dice coefficient"""
        intersection = (predictions & targets).float().sum()
        total = predictions.float().sum() + targets.float().sum()
        
        if total == 0:
            return 1.0
        
        return (2.0 * intersection / total).item()


class EvaluatorFactory:
    """Factory to create appropriate evaluator"""
    
    @staticmethod
    def create_evaluator(task_type: str, model_ref: ModelReference, 
                        num_classes: int, **kwargs) -> BaseEvaluator:
        """Create evaluator based on task type"""
        task_type = task_type.lower()
        
        if task_type in ['classification', 'image_classification']:
            return ClassificationEvaluator(model_ref, num_classes, **kwargs)
        elif task_type in ['segmentation', 'image_segmentation']:
            return SegmentationEvaluator(model_ref, num_classes, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")


# Example usage
if __name__ == "__main__":
    # Create model reference
    model_ref = ModelReference(
        model_name="resnet50_classifier",
        model_path="./models/resnet50.pth",
        model_type="classification",
        architecture="ResNet50"
    )
    
    # Create evaluator
    evaluator = EvaluatorFactory.create_evaluator(
        task_type='classification',
        model_ref=model_ref,
        num_classes=10,
        device='cuda',
        batch_size=32
    )
    
    print(f"Created evaluator for {model_ref.model_name}")
    print(f"Tracker save directory: {evaluator.tracker.save_dir}")
    
    # Example of tracking training progress
    train_metrics = {'loss': 0.5, 'accuracy': 0.85}
    val_metrics = {'loss': 0.6, 'accuracy': 0.82}
    evaluator.track_training_progress(train_metrics, val_metrics, step=100)
    
    # End epoch
    evaluator.end_epoch()
    
    print("Example evaluator created successfully!")