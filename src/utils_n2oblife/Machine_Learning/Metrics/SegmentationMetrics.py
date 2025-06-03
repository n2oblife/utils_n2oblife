import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from skimage.transform import resize
from sklearn.metrics import roc_auc_score, jaccard_score, precision_recall_curve, auc
import warnings
from dataclasses import dataclass, field

# Import the enhanced Metrics class
from utils_n2oblife.Machine_Learning.Metrics.Metrics import Metrics, EvaluationMetrics, metric_catcher

EVAL_METHODS = [
    'AUC', 'IoU_quantitative', 'IoU_qualitative', 'Dice', 'Precision', 'Recall', 
    'F1', 'Specificity', 'Sensitivity', 'Hausdorff', 'Surface_Distance', 'PR_AUC'
]

@dataclass
class SegmentationResults:
    """Container for segmentation evaluation results"""
    quantitative_metrics: Dict[str, float] = field(default_factory=dict)
    qualitative_metrics: Dict[str, Union[float, np.ndarray]] = field(default_factory=dict)
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_evaluation_metrics(self, loss: float = 0.0, inference_time: float = 0.0) -> EvaluationMetrics:
        """Convert to EvaluationMetrics format"""
        return EvaluationMetrics(
            accuracy=self.quantitative_metrics.get('Accuracy', 0.0),
            precision=self.quantitative_metrics.get('Precision', 0.0),
            recall=self.quantitative_metrics.get('Recall', 0.0),    
            f1_score=self.quantitative_metrics.get('F1', 0.0),
            iou=self.quantitative_metrics.get('IoU_quantitative', 0.0),
            dice_coefficient=self.quantitative_metrics.get('Dice', 0.0),
            loss=loss,
            inference_time=inference_time,
            custom_metrics={
                **{k: v for k, v in self.quantitative_metrics.items() 
                   if k not in ['Accuracy', 'Precision', 'Recall', 'F1', 'IoU_quantitative', 'Dice']},
                **{f'qualitative_{k}': v for k, v in self.qualitative_metrics.items() 
                   if isinstance(v, (int, float))}
            }
        )

class SegmentationMetrics2D(Metrics):
    """Enhanced 2D Segmentation Metrics Class"""
    
    def __init__(self, 
                 predicted: Optional[np.ndarray] = None, 
                 target: Optional[np.ndarray] = None, 
                 threshold: float = 0.5, 
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 requested_metrics: Optional[List[str]] = None,
                 ignore_background: bool = True,
                 spacing: Optional[Tuple[float, ...]] = None) -> None:
        """
        Initialize SegmentationMetrics2D
        
        Args:
            predicted: Predicted segmentation mask(s)
            target: Ground truth segmentation mask(s)  
            threshold: Threshold for binary classification (default: 0.5)
            multi_class: Whether this is multi-class segmentation
            num_classes: Number of classes for multi-class segmentation
            requested_metrics: List of metrics to compute
            ignore_background: Whether to ignore background class in multi-class
            spacing: Physical spacing for distance metrics (z, y, x)
        """
        # Initialize parent class with requested metrics
        super().__init__(requested_metrics=requested_metrics or EVAL_METHODS)
        
        self.predicted = predicted
        self.target = target
        self.threshold = threshold
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.spacing = spacing or (1.0, 1.0, 1.0)
        
        # Storage for results
        self.results = SegmentationResults()
        self._processed_predictions = None
        self._processed_targets = None
        
        # Validate inputs if provided
        if predicted is not None and target is not None:
            self._validate_inputs()
    
    def load_images(self, predicted: Optional[np.ndarray] = None, 
                   target: Optional[np.ndarray] = None) -> None:
        """Load new prediction and target images"""
        if predicted is not None:
            self.predicted = predicted
        if target is not None:
            self.target = target
        
        # Reset processed data
        self._processed_predictions = None
        self._processed_targets = None
        
        if self.predicted is not None and self.target is not None:
            self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input arrays"""
        if self.predicted is None:
            raise ValueError("Predicted mask is required")
        if self.target is None:
            raise ValueError("Target mask is required")
        
        # Convert to numpy arrays if needed
        self.predicted = np.asarray(self.predicted)
        self.target = np.asarray(self.target)
        
        # Check dimensions
        if self.predicted.ndim != self.target.ndim:
            raise ValueError(f"Predicted and target must have same number of dimensions. "
                           f"Got {self.predicted.ndim} and {self.target.ndim}")
        
        if self.predicted.ndim not in [2, 3, 4]:  # 2D, 3D, or batch of 2D/3D
            raise ValueError(f"Expected 2D, 3D, or batched arrays, got {self.predicted.ndim}D")
    
    def _preprocess_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess inputs for computation"""
        if self._processed_predictions is not None and self._processed_targets is not None:
            return self._processed_predictions, self._processed_targets
        
        predicted = self.predicted.copy()
        target = self.target.copy()
        
        # Resize if shapes don't match
        if predicted.shape != target.shape:
            warnings.warn(f"Resizing predicted from {predicted.shape} to {target.shape}")
            predicted = resize(predicted, target.shape, preserve_range=True, anti_aliasing=False)
        
        # Handle multi-class vs binary
        if self.multi_class:
            # Ensure integer labels for multi-class
            target = target.astype(np.int32)
            if predicted.shape[-1] == self.num_classes:  # One-hot or softmax output
                predicted = np.argmax(predicted, axis=-1).astype(np.int32)
            else:
                predicted = predicted.astype(np.int32)
        else:
            # Binary segmentation - apply threshold
            if predicted.dtype != bool:
                predicted = predicted > self.threshold
            target = target.astype(bool)
        
        self._processed_predictions = predicted
        self._processed_targets = target
        
        return predicted, target
    
    def _compute_confusion_matrix(self, predicted: np.ndarray, target: np.ndarray) -> Dict[str, int]:
        """Compute confusion matrix elements"""
        if self.multi_class:
            # For multi-class, compute per-class confusion matrix
            # This is simplified - you might want to use sklearn.metrics.confusion_matrix
            tp = np.sum((predicted == target) & (target > 0))  # Correct non-background
            fp = np.sum((predicted != target) & (predicted > 0))  # False positive
            fn = np.sum((predicted != target) & (target > 0))    # False negative
            tn = np.sum((predicted == target) & (target == 0))   # True negative (background)
        else:
            # Binary segmentation
            tp = np.sum(predicted & target)
            fp = np.sum(predicted & ~target)
            fn = np.sum(~predicted & target)
            tn = np.sum(~predicted & ~target)
        
        return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
    
    @metric_catcher
    def compute_AUC(self) -> float:
        """Compute Area Under ROC Curve"""
        if not self._should_compute('AUC'):
            return None
            
        predicted, target = self._preprocess_inputs()
        
        try:
            if self.multi_class:
                # For multi-class, compute macro-averaged AUC
                aucs = []
                for class_id in range(1, self.num_classes):  # Skip background if ignore_background
                    class_target = (target == class_id).astype(int)
                    class_pred = (predicted == class_id).astype(float)
                    if np.sum(class_target) > 0:  # Only if class exists
                        auc_score = roc_auc_score(class_target.flatten(), class_pred.flatten())
                        aucs.append(auc_score)
                auc = np.mean(aucs) if aucs else 0.0
            else:
                # Use original predicted probabilities if available, otherwise use binary
                pred_probs = self.predicted if self.predicted.dtype != bool else predicted.astype(float)
                auc = roc_auc_score(target.flatten(), pred_probs.flatten())
            
            self.results.quantitative_metrics['AUC'] = auc
            return auc
        except ValueError as e:
            warnings.warn(f"Could not compute AUC: {e}")
            return 0.0
    
    @metric_catcher
    def compute_PR_AUC(self) -> float:
        """Compute Area Under Precision-Recall Curve"""
        if not self._should_compute('PR_AUC'):
            return None
            
        predicted, target = self._preprocess_inputs()
        
        try:
            if self.multi_class:
                # Simplified for multi-class
                pr_aucs = []
                for class_id in range(1, self.num_classes):
                    class_target = (target == class_id).astype(int)
                    class_pred = (predicted == class_id).astype(float)
                    if np.sum(class_target) > 0:
                        precision, recall, _ = precision_recall_curve(
                            class_target.flatten(), class_pred.flatten()
                        )
                        pr_auc = auc(recall, precision)
                        pr_aucs.append(pr_auc)
                pr_auc_score = np.mean(pr_aucs) if pr_aucs else 0.0
            else:
                pred_probs = self.predicted if self.predicted.dtype != bool else predicted.astype(float)
                precision, recall, _ = precision_recall_curve(
                    target.flatten(), pred_probs.flatten()
                )
                pr_auc_score = auc(recall, precision)
            
            self.results.quantitative_metrics['PR_AUC'] = pr_auc_score 
            return pr_auc_score
        except ValueError as e:
            warnings.warn(f"Could not compute PR-AUC: {e}")
            return 0.0
    
    @metric_catcher
    def compute_IoU_quantitative(self) -> float:
        """Compute quantitative IoU (Jaccard Index)"""
        if not self._should_compute('IoU_quantitative'):
            return None
            
        predicted, target = self._preprocess_inputs()
        
        try:
            if self.multi_class:
                # Use sklearn's jaccard_score with appropriate averaging
                iou = jaccard_score(
                    target.flatten(), 
                    predicted.flatten(), 
                    average='macro',
                    zero_division=0
                )
            else:
                # Manual computation for binary case
                intersection = np.sum(predicted & target)
                union = np.sum(predicted | target)
                iou = intersection / union if union > 0 else 0.0
            
            self.results.quantitative_metrics['IoU_quantitative'] = iou
            return iou
        except Exception as e:
            warnings.warn(f"Could not compute IoU: {e}")
            return 0.0
    
    @metric_catcher
    def compute_IoU_qualitative(self) -> np.ndarray:
        """Compute per-class IoU"""
        if not self._should_compute('IoU_qualitative'):
            return None
            
        predicted, target = self._preprocess_inputs()
        
        try:
            if self.multi_class:
                iou_per_class = jaccard_score(
                    target.flatten(), 
                    predicted.flatten(), 
                    average=None,
                    zero_division=0
                )
            else:
                # For binary, return single value as array
                intersection = np.sum(predicted & target)
                union = np.sum(predicted | target)
                iou_per_class = np.array([intersection / union if union > 0 else 0.0])
            
            self.results.qualitative_metrics['IoU_qualitative'] = iou_per_class
            return iou_per_class
        except Exception as e:
            warnings.warn(f"Could not compute per-class IoU: {e}")
            return np.array([0.0])
    
    @metric_catcher
    def compute_Dice(self) -> float:
        """Compute Dice Coefficient"""
        if not self._should_compute('Dice'):
            return None
            
        predicted, target = self._preprocess_inputs()
        
        if self.multi_class:
            # Compute mean Dice across all classes
            dice_scores = []
            for class_id in range(self.num_classes):
                if self.ignore_background and class_id == 0:
                    continue
                class_pred = (predicted == class_id)
                class_target = (target == class_id)
                
                intersection = np.sum(class_pred & class_target)
                total = np.sum(class_pred) + np.sum(class_target)
                
                if total > 0:
                    dice = 2.0 * intersection / total
                    dice_scores.append(dice)
            
            dice = np.mean(dice_scores) if dice_scores else 0.0
        else:
            # Binary Dice
            intersection = np.sum(predicted & target)
            total = np.sum(predicted) + np.sum(target)
            dice = 2.0 * intersection / total if total > 0 else 0.0
        
        self.results.quantitative_metrics['Dice'] = dice
        return dice
    
    @metric_catcher
    def compute_segmentation_metrics(self) -> Dict[str, float]:
        """Compute standard segmentation metrics (Precision, Recall, F1, etc.)"""
        predicted, target = self._preprocess_inputs()
        cm = self._compute_confusion_matrix(predicted, target)
        
        metrics = {}
        
        # Basic metrics
        tp, fp, fn, tn = cm['TP'], cm['FP'], cm['FN'], cm['TN']
        
        # Precision (PPV)
        if self._should_compute('Precision'):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['Precision'] = precision
            self.results.quantitative_metrics['Precision'] = precision
        
        # Recall (Sensitivity, TPR)
        if self._should_compute('Recall') or self._should_compute('Sensitivity'):
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['Recall'] = recall
            metrics['Sensitivity'] = recall
            self.results.quantitative_metrics['Recall'] = recall
            self.results.quantitative_metrics['Sensitivity'] = recall
        
        # Specificity (TNR)
        if self._should_compute('Specificity'):
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['Specificity'] = specificity
            self.results.quantitative_metrics['Specificity'] = specificity
        
        # F1 Score
        if self._should_compute('F1'):
            precision = metrics.get('Precision', tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recall = metrics.get('Recall', tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics['F1'] = f1
            self.results.quantitative_metrics['F1'] = f1
        
        # Accuracy
        if self._should_compute('Accuracy'):
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            metrics['Accuracy'] = accuracy
            self.results.quantitative_metrics['Accuracy'] = accuracy
        
        return metrics
    
    def evaluate_model(self, verbose: bool = True) -> SegmentationResults:
        """Evaluate model with all requested metrics"""
        if self.predicted is None or self.target is None:
            raise ValueError("Both predicted and target masks must be provided")
        
        try:
            # Compute all requested metrics
            if 'AUC' in self.requested_metrics:
                self.compute_AUC()
            
            if 'PR_AUC' in self.requested_metrics:
                self.compute_PR_AUC()
            
            if 'IoU_quantitative' in self.requested_metrics:
                self.compute_IoU_quantitative()
            
            if 'IoU_qualitative' in self.requested_metrics:
                self.compute_IoU_qualitative()
            
            if 'Dice' in self.requested_metrics:
                self.compute_Dice()
            
            # Compute standard metrics
            standard_metrics = ['Precision', 'Recall', 'F1', 'Specificity', 'Sensitivity', 'Accuracy']
            if any(metric in self.requested_metrics for metric in standard_metrics):
                self.compute_segmentation_metrics()
            
            if verbose:
                self._print_results()
            
            return self.results
            
        except Exception as e:
            warnings.warn(f"Error during evaluation: {e}")
            return self.results
    
    def _print_results(self) -> None:
        """Print evaluation results"""
        print("\n" + "="*50)
        print("SEGMENTATION EVALUATION RESULTS")
        print("="*50)
        
        if self.results.quantitative_metrics:
            print("\nQuantitative Metrics:")
            for metric, value in self.results.quantitative_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        if self.results.qualitative_metrics:
            print("\nQualitative Metrics:")
            for metric, value in self.results.qualitative_metrics.items():
                if isinstance(value, np.ndarray):
                    print(f"  {metric}: {value}")
                else:
                    print(f"  {metric}: {value:.4f}")
    
    def to_evaluation_metrics(self, loss: float = 0.0, inference_time: float = 0.0) -> EvaluationMetrics:
        """Convert results to EvaluationMetrics format"""
        return self.results.to_evaluation_metrics(loss, inference_time)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    target = np.random.randint(0, 2, (128, 128)).astype(bool)
    predicted = target.copy()
    # Add some noise
    noise_mask = np.random.random((128, 128)) < 0.1
    predicted[noise_mask] = ~predicted[noise_mask]
    
    # Initialize metrics
    seg_metrics = SegmentationMetrics2D(
        predicted=predicted,
        target=target,
        threshold=0.5,
        requested_metrics=['AUC', 'IoU_quantitative', 'Dice', 'F1', 'Precision', 'Recall']
    )
    
    # Evaluate
    results = seg_metrics.evaluate_model()
    
    # Convert to EvaluationMetrics
    eval_metrics = seg_metrics.to_evaluation_metrics(loss=0.1, inference_time=0.05)
    print(f"\nEvaluationMetrics dict: {eval_metrics.to_dict()}")