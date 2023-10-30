from .ClassicMetrics import Metrics

EVAL_METHOD = ['AUC', 'IoU_quantitative', 'IoU_qualitative']

# already implemented in tensorflow
class SegmentationMetrics2D(Metrics):
    def __init__(self, predicted, target, threshold = 0., multi_mask=False) -> None:
        super().__init__()
        self.predicted = predicted
        self.target = target
        self.threshold = threshold
        self.multi_mask = multi_mask
        self.all_predicted_masks = {}
        self.quali_eval = {}
        self.quant_eval = {}
    
    def load_images(self, predicted=None, target=None)->None:
        if predicted:
            self.predicted = predicted
        if target:
            self.target = target
