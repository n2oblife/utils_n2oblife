#This class is an implementation of some metrics 
#some can be seen here : https://en.wikipedia.org/wiki/Sensitivity_and_specificity

import time
import functools
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from Evaluate import EvaluationMetrics

METRICS = ['P','N','TP', 'TN', 'FP', 'FN', 
		   'TPR','FNR', 'TNR', 'FPR',
		   'PPV', 'FDR', 'NPV', 'FOR', 'LR+', 'LR-', 'PT', 'TS',
		   'prev', 'ACC', 'BA', 'F1', 'IOU', 'DICE']

def metric_catcher(f):
	@functools.wraps(f)
	def func(*args, **kwargs):
		try:
			return f(*args, **kwargs)
		except Exception as e:
			print(f'Need to initiate the metrics correctly and save them: {e}')
			return None
	return func

class Metrics:
    """Enhanced Metrics class compatible with EvaluationMetrics"""
    
    def __init__(self, requested_metrics: List[str] = None, P=0, N=0, TP=0, TN=0, FP=0, FN=0) -> None:
        """Initialize metrics class with only requested metrics loaded
        
        Args:
            requested_metrics (List[str], optional): List of metrics to compute. Defaults to all.
            P (int, optional): Positive. Defaults to 0.
            N (int, optional): Negative. Defaults to 0.
            TP (int, optional): True Positive. Defaults to 0.
            TN (int, optional): True Negative. Defaults to 0.
            FP (int, optional): False Positive. Defaults to 0.
            FN (int, optional): False Negative. Defaults to 0.
        """
        # Base confusion matrix values
        self.base_metrics = {'P': P, 'N': N, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
        
        # Store requested metrics (if None, compute all)
        self.requested_metrics = requested_metrics or METRICS.copy()
        
        # Initialize computed metrics storage
        self.computed_metrics = {}
        
        # Mapping between metric names and EvaluationMetrics fields
        self.metric_mapping = {
            'ACC': 'accuracy',
            'PPV': 'precision', 
            'TPR': 'recall',
            'F1': 'f1_score',
            'IOU': 'iou',
            'DICE': 'dice_coefficient'
        }
    
    def _save_init_metrics(self, P=0, N=0, TP=0, TN=0, FP=0, FN=0) -> None:
        """Update base metrics with new values
        
        Args:
            P (int, optional): Positive. Defaults to 0.
            N (int, optional): Negative. Defaults to 0.
            TP (int, optional): True Positive. Defaults to 0.
            TN (int, optional): True Negative. Defaults to 0.
            FP (int, optional): False Positive. Defaults to 0.
            FN (int, optional): False Negative. Defaults to 0.
        """
        args = {'P': P, 'N': N, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
        for key, value in args.items():
            if self.base_metrics[key] == 0:
                self.base_metrics[key] = value
    
    def _save(self, metric_key: str, value: Any) -> None:
        """Save computed metric if it's in requested metrics
        
        Args:
            metric_key (str): the key of the metric
            value (any): the value of the metric
        """
        if metric_key in METRICS and metric_key in self.requested_metrics:
            self.computed_metrics[metric_key] = value
        elif metric_key not in METRICS:
            raise ValueError(f'The metric to save must be among {METRICS}')
    
    def _should_compute(self, metric_key: str) -> bool:
        """Check if metric should be computed based on requested metrics"""
        return metric_key in self.requested_metrics
    
    @metric_catcher
    def compute_TPR(self, TP=None, FN=None) -> float:
        """Compute True Positive Rate (Recall/Sensitivity)"""
        if not self._should_compute('TPR'):
            return None
            
        TP = TP if TP is not None else self.base_metrics['TP']
        FN = FN if FN is not None else self.base_metrics['FN']
        
        if TP + FN == 0:
            return 0.0
        result = TP / (TP + FN)
        self._save('TPR', result)
        return result

    @metric_catcher
    def compute_TNR(self, TN=None, N=None) -> float:
        """Compute True Negative Rate (Specificity)"""
        if not self._should_compute('TNR'):
            return None
            
        TN = TN if TN is not None else self.base_metrics['TN']
        N = N if N is not None else self.base_metrics['N']
        
        if N == 0:
            return 0.0
        result = TN / N
        self._save('TNR', result)
        return result

    @metric_catcher
    def compute_PPV(self, TP=None, FP=None) -> float:
        """Compute Positive Predictive Value (Precision)"""
        if not self._should_compute('PPV'):
            return None
            
        TP = TP if TP is not None else self.base_metrics['TP']
        FP = FP if FP is not None else self.base_metrics['FP']
        
        if TP + FP == 0:
            return 0.0
        result = TP / (TP + FP)
        self._save('PPV', result)
        return result

    @metric_catcher	
    def compute_NPV(self, TN=None, FN=None) -> float:
        """Compute Negative Predictive Value"""
        if not self._should_compute('NPV'):
            return None
            
        TN = TN if TN is not None else self.base_metrics['TN']
        FN = FN if FN is not None else self.base_metrics['FN']
        
        if TN + FN == 0:
            return 0.0
        result = TN / (TN + FN)
        self._save('NPV', result)
        return result
    
    @metric_catcher
    def compute_FNR(self, FN=None, P=None) -> float:
        """Compute False Negative Rate"""
        if not self._should_compute('FNR'):
            return None
            
        FN = FN if FN is not None else self.base_metrics['FN']
        P = P if P is not None else self.base_metrics['P']
        
        if P == 0:
            return 0.0
        result = FN / P
        self._save('FNR', result)
        return result
    
    @metric_catcher
    def compute_FPR(self, FP=None, N=None) -> float:
        """Compute False Positive Rate"""
        if not self._should_compute('FPR'):
            return None
            
        FP = FP if FP is not None else self.base_metrics['FP']
        N = N if N is not None else self.base_metrics['N']
        
        if N == 0:
            return 0.0
        result = FP / N
        self._save('FPR', result)
        return result
    
    @metric_catcher
    def compute_ACC(self, TP=None, TN=None, P=None, N=None) -> float:
        """Compute Accuracy"""
        if not self._should_compute('ACC'):
            return None
            
        TP = TP if TP is not None else self.base_metrics['TP']
        TN = TN if TN is not None else self.base_metrics['TN']
        P = P if P is not None else self.base_metrics['P']
        N = N if N is not None else self.base_metrics['N']
        
        total = P + N
        if total == 0:
            return 0.0
        result = (TP + TN) / total
        self._save('ACC', result)
        return result
    
    @metric_catcher
    def compute_F1_score(self, PPV=None, TPR=None) -> float:
        """Compute F1 Score"""
        if not self._should_compute('F1'):
            return None
            
        if PPV is None:
            PPV = self.compute_PPV()
        if TPR is None:
            TPR = self.compute_TPR()
            
        if PPV is None or TPR is None or (PPV + TPR) == 0:
            return 0.0
        result = 2 * PPV * TPR / (PPV + TPR)
        self._save('F1', result)
        return result
    
    @metric_catcher
    def compute_IOU(self, TP=None, FP=None, FN=None) -> float:
        """Compute Intersection over Union (Jaccard Index)"""
        if not self._should_compute('IOU'):
            return None
            
        TP = TP if TP is not None else self.base_metrics['TP']
        FP = FP if FP is not None else self.base_metrics['FP']
        FN = FN if FN is not None else self.base_metrics['FN']
        
        union = TP + FP + FN
        if union == 0:
            return 0.0
        result = TP / union
        self._save('IOU', result)
        return result
    
    @metric_catcher
    def compute_DICE(self, TP=None, FP=None, FN=None) -> float:
        """Compute Dice Coefficient (F1 Score for segmentation)"""
        if not self._should_compute('DICE'):
            return None
            
        TP = TP if TP is not None else self.base_metrics['TP']
        FP = FP if FP is not None else self.base_metrics['FP']
        FN = FN if FN is not None else self.base_metrics['FN']
        
        denominator = 2 * TP + FP + FN
        if denominator == 0:
            return 0.0
        result = (2 * TP) / denominator
        self._save('DICE', result)
        return result
    
    def compute_all_requested(self) -> Dict[str, float]:
        """Compute all requested metrics and return as dictionary"""
        results = {}
        
        # Compute basic metrics
        if 'TPR' in self.requested_metrics:
            results['TPR'] = self.compute_TPR()
        if 'TNR' in self.requested_metrics:
            results['TNR'] = self.compute_TNR()
        if 'PPV' in self.requested_metrics:
            results['PPV'] = self.compute_PPV()
        if 'NPV' in self.requested_metrics:
            results['NPV'] = self.compute_NPV()
        if 'ACC' in self.requested_metrics:
            results['ACC'] = self.compute_ACC()
        if 'F1' in self.requested_metrics:
            results['F1'] = self.compute_F1_score()
        if 'IOU' in self.requested_metrics:
            results['IOU'] = self.compute_IOU()
        if 'DICE' in self.requested_metrics:
            results['DICE'] = self.compute_DICE()
        
        # Filter out None values
        return {k: v for k, v in results.items() if v is not None}
    
    def to_evaluation_metrics(self, loss: float = 0.0, inference_time: float = 0.0, 
                            custom_metrics: Dict[str, float] = None) -> EvaluationMetrics:
        """Convert computed metrics to EvaluationMetrics dataclass
        
        Args:
            loss (float): Loss value to include
            inference_time (float): Inference time to include
            custom_metrics (Dict[str, float]): Additional custom metrics
            
        Returns:
            EvaluationMetrics: Populated dataclass
        """
        # Compute all requested metrics
        computed = self.compute_all_requested()
        
        # Create EvaluationMetrics with mapped values
        eval_metrics = EvaluationMetrics(
            accuracy=computed.get('ACC', 0.0),
            precision=computed.get('PPV', 0.0),
            recall=computed.get('TPR', 0.0),
            f1_score=computed.get('F1', 0.0),
            iou=computed.get('IOU', 0.0),
            dice_coefficient=computed.get('DICE', 0.0),
            loss=loss,
            inference_time=inference_time,
            custom_metrics=custom_metrics or {}
        )
        
        # Add any additional computed metrics to custom_metrics
        for metric_key, value in computed.items():
            if metric_key not in ['ACC', 'PPV', 'TPR', 'F1', 'IOU', 'DICE']:
                eval_metrics.custom_metrics[metric_key] = value
        
        return eval_metrics
    
    # Keep all original methods for backward compatibility
    @metric_catcher
    def compute_FDR(self, FP=None, TP=None) -> float:
        """Compute False Discovery Rate"""
        FP = FP if FP is not None else self.base_metrics['FP']
        TP = TP if TP is not None else self.base_metrics['TP']
        
        if FP + TP == 0:
            return 0.0
        return FP / (FP + TP)
	
    @metric_catcher
    def compute_FOR(self, FN=None, TN=None) -> float:
        """Compute False Omission Rate"""
        FN = FN if FN is not None else self.base_metrics['FN']
        TN = TN if TN is not None else self.base_metrics['TN']
        
        if FN + TN == 0:
            return 0.0
        return FN / (FN + TN)
    
    @metric_catcher
    def compute_prevalence(self, P=None, N=None) -> float:
        """Compute Prevalence"""
        P = P if P is not None else self.base_metrics['P']
        N = N if N is not None else self.base_metrics['N']
        
        if P + N == 0:
            return 0.0
        return P / (P + N)


# Example usage
if __name__ == "__main__":
    # Initialize with only specific metrics
    metrics = Metrics(
        requested_metrics=['ACC', 'PPV', 'TPR', 'F1', 'IOU', 'DICE'],
        TP=85, TN=90, FP=10, FN=15, P=100, N=100
    )
    
    # Convert to EvaluationMetrics
    eval_metrics = metrics.to_evaluation_metrics(
        loss=0.15, 
        inference_time=0.025,
        custom_metrics={'custom_score': 0.92}
    )
    
    print("EvaluationMetrics:")
    print(eval_metrics.to_dict())