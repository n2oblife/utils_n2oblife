import numpy as np
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from utils_n2oblife.Machine_Learning.Metrics.ClassicMetrics import Metrics

#TODO import eval methods
EVAL_METHOD = ['AUC', 'IoU']

class SegmentationMetrics2D(Metrics):
    def __init__(self, threshold = 0., multiple_masks = False) -> None :
        super().__init__(self)
        self.threshold = threshold
        self.all_predicted_masks = {}
        self.qualitative_eval = {}
        self.all_eval = {}

    def save(self, id='', masks = [], TP=0, FP=0, FN=0, AP=0, AUC=0) -> None :
        self.all_predicted_masks[id] = masks
        self.qualitative_eval[id] = {'TP': TP, 'FP': FP, 'FN': FN, 'AP': AP, 'AUC': AUC}

    def compute_AUC(self, predicted = None, target = None) -> list[float]:
        assert(predicted is not None), ValueError("Need to give a predicted input")
        assert(target is not None), ValueError("Need to give a target input")

        # resize the inputs if necessary
        if predicted.shape != target.shape:
            predicted = resize(predicted, target.shape, preserve_range=True)
            target = resize(target, target.shape, preserve_range=True)

        pix_predicted = np.sum(predicted)
        pix_ground_truth = np.sum(target)
        pix_AUC = np.sum(np.logical_and(predicted, target))

        surface = predicted.size
        return pix_AUC/surface, pix_predicted/surface, pix_ground_truth/surface

    def harmonizing_size(self, predicted, target):
        if predicted.shape != target.shape:
            predicted = resize(predicted, target.shape, preserve_range=True)
            target = resize(target, target.shape, preserve_range=True)
        return predicted, target

    def positive_sample_matching_pix(self):
        #TODO based on Fast and accurate cable detection using CNN
        # or compare to feature map
        pass

    def prediction_correct_pix(self, AUC=0, predicted=0, target=0, method = ''):
        assert(not(AUC==0 and predicted==0 and target==0)), ValueError('All the input must be non null')
        assert (method in EVAL_METHOD), ValueError(f'the evaluation method is not in {EVAL_METHOD}')

        if method == 'AUC':
            to_compare = AUC
        elif method == 'diff':
            to_compare = predicted

        rest = (target - to_compare)**2
        return rest <= self.pix_threshold

    def evaluate_model(self):
        for id, masks in self.all_predicted_masks.items():
            for mask in masks:
                _, predicted, ground_truth = self.compute_AUC(mask, self.qualitative_eval[id]['ground_truth'])
                self.all_eval[id] = {'AUC': roc_auc_score(ground_truth.flatten(), predicted.flatten())}
