import numpy as np
from skimage.transform import resize
from sklearn.metrics import roc_auc_score, jaccard_score
from utils_n2oblife.Machine_Learning.Metrics.ClassicMetrics import Metrics


EVAL_METHOD = ['AUC', 'IoU_quantitative', 'IoU_qualitative']

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
        if predicted is not None:
            self.predicted = predicted
        if target is not None:
            self.target = target

    def compute_AUC(self):
        assert(self.predicted is not None), ValueError("Need to give a predicted input")
        assert(self.target is not None), ValueError("Need to give a target input")

        # resize the inputs if necessary
        if self.predicted.shape != self.target.shape:
            self.predicted = resize(self.predicted, self.target.shape, preserve_range=True)
            self.target = resize(self.target, self.target.shape, preserve_range=True)

        auc = roc_auc_score(self.target.flatten(), self.predicted.flatten())
        return auc

    def compute_IoU_quantitative(self):
        assert(self.predicted is not None), ValueError("Need to give a predicted input")
        assert(self.target is not None), ValueError("Need to give a target input")

        # resize the inputs if necessary
        if self.predicted.shape != self.target.shape:
            self.predicted = resize(self.predicted, self.target.shape, preserve_range=True)
            self.target = resize(self.target, self.target.shape, preserve_range=True)

        iou = jaccard_score(self.target.flatten(), self.predicted.flatten(), average='weighted')
        return iou

    def compute_IoU_qualitative(self):
        assert(self.predicted is not None), ValueError("Need to give a predicted input")
        assert(self.target is not None), ValueError("Need to give a target input")

        # resize the inputs if necessary
        if self.predicted.shape != self.target.shape:
            self.predicted = resize(self.predicted, self.target.shape, preserve_range=True)
            self.target = resize(self.target, self.target.shape, preserve_range=True)

        iou = jaccard_score(self.target.flatten(), self.predicted.flatten(), average=None)
        return iou

    def evaluate_model(self):
        self.quant_eval['AUC'] = self.compute_AUC()
        self.quant_eval['IoU_quantitative'] = self.compute_IoU_quantitative()
        self.quali_eval['IoU_qualitative'] = self.compute_IoU_qualitative()
        return self.quant_eval, self.quali_eval
