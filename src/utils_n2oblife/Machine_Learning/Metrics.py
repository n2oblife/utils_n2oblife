#This class is an implementation of some metrics 
#some can be seen here : https://en.wikipedia.org/wiki/Sensitivity_and_specificity

import functools
import numpy as np

METRICS = ['P','N','TP', 'TN', 'FP', 'FN', 'TPR','FNR', 'TNR', 'FPR',
		   'PPV', 'FDR', 'NPV', 'FOR', 'LR+', 'LR-', 'PT', 'TS',
		   'prev', 'ACC', 'BA', 'F1']

EVAL_METHOD_PIX = ['AUC', 'diff' ]

def metric_catcher(f):
	@functools.wraps(f)
	def func(*args, **kwargs):
		try:
			return f(*args, **kwargs)
		except Exception as e:
			print(f'Need to initiate the metrics correctly and save them')
	return func

class Metrics():
	def __init__(self, P=0, N=0, TP=0, TN=0, FP=0, FN=0) -> None:
		"""a class template to be adapted for any kind of task 
		(for now specific to ML)

		Args:
			P (int, optional): Positive. Defaults to 0.
			N (int, optional): Negative. Defaults to 0.
			TP (int, optional): True Psotive. Defaults to 0.
			TN (int, optional): True Negative. Defaults to 0.
			FP (int, optional): False Positive. Defaults to 0.
			FN (int, optional): False NEgative. Defaults to 0.
		"""
		# TODO maybe initiate the values time at a time and see if they need to be called in the functions
		self.metrics = {'P':P,'N':N,
				  'TP':TP, 'TN':TN,
				  'FP':FP, 'FN':FN}
	
	def _save_init_metrics(self, P=0, N=0, TP=0, TN=0, FP=0, FN=0) -> None:
		"""Initiate the init metrics with good values

		Args:
			P (int, optional): Positive. Defaults to 0.
			N (int, optional): Negative. Defaults to 0.
			TP (int, optional): True Psotive. Defaults to 0.
			TN (int, optional): True Negative. Defaults to 0.
			FP (int, optional): False Positive. Defaults to 0.
			FN (int, optional): False NEgative. Defaults to 0.
		"""
		args = {'P':P, 'N':N, 'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
		for key, value in args:
			if self.metrics[key] == 0 :
				self.metrics[key] = value

	def _save(self, metric_key:str, value:any)->None:
		"""Saves metrics that are of the good type

		Args:
			metric_key (str): the key of the metric
			value (any): the value of the metric
		"""
		if metric_key in METRICS:
			self.metrics[metric_key] = value
		else:
			raise ValueError(f'The metric to save must be among {self.metrics_types}')
	
	def compute_mean_distance(self, predicted = None, target = None, power = 1, n_elts = 2) -> float:
		"""This function computes the mean distance, 
		still need work to be very generic

		Args:
			predicted (any, optional): _description_. Defaults to None.
			target (any, optional): _description_. Defaults to None.
			power (int, optional): _description_. Defaults to 1.
			n_elts (int, optional): _description_. Defaults to 2.

		Returns:
			float: mean distance
		"""
		return NotImplemented

	@metric_catcher
	def compute_TPR(self, TP = 0, FN = 0) -> float:
		"""This function computes the recall or the true positive rate

		Args:
			TP (int, optional): number of True Positive. Defaults to 0.
			FN (int, optional): number of False Negative. Defaults to 0.

		Returns:
			float: TPR
		"""
		assert((TP !=0 and FN != 0) or TP >= 0 or FN >= 0), ValueError(
		"At least True positive or False negative must be non null and positive")
		return TP / (TP + FN)

	@metric_catcher
	def compute_TNR(self,TN=0, N=0):
		"""This function computes the specifity or the true negative rate

		Args:
			TN (int, optional): number of True Positive. Defaults to 0.
			N (int, optional): number of Negative. Defaults to 0.
		
		Returns:
			float: TNR
		"""
		assert(N>0 or TN>=0), ValueError('There must be negative values to compute this rate and they must be positive')
		return TN/N

	@metric_catcher
	def compute_PPV(self, TP = 0, FP = 0) -> float:
		"""This function computes the precision or the positive prediction value

		Args:
			TP (int, optional): number of True Positive. Defaults to 0.
			FP (int, optional): number of False Positive. Defaults to 0.

		Returns:
			float: PPV
		"""
		assert((TP !=0 and FP != 0) or TP >= 0 or FP >= 0), ValueError(
		"At least True positive or False positive must be non null and positive")
		return TP / (TP + FP)

	@metric_catcher	
	def compute_NPV(self, TN=0, FN=0):
		"""This function computes the precision or the negative prediction value

		Args:
			TN (int, optional): number of True Negative. Defaults to 0.
			FN (int, optional): number of False Negative. Defaults to 0.

		Returns:
			float: NPV
		"""
		assert((TN !=0 and FN != 0) or TN >= 0 or FN >= 0), ValueError(
		"At least True negative or False negative must be non null and positive")
		return TN/(TN+FN)
	
	@metric_catcher
	def compute_FNR(self, FN=0, P=0):
		"""
		This function computes the False Negative Rate (FNR), also known as the Miss Rate.

		Args:
			FN (int, optional): Number of False Negatives. Defaults to 0.
			P (int, optional): Number of Positive. Defaults to 0.

		Returns:
			float: FNR
		"""
		assert(FN>=0 or P>0), ValueError(
			'At least the positive must be non null and false negative more than 0')
		return FN/P
	
	@metric_catcher
	def compute_FPR(self, FP=0, N=0):
		"""
		This function computes the False Psotive Rate (FPR), also known as the Fall out.

		Args:
			FP (int, optional): Number of False Positive. Defaults to 0.
			N (int, optional): Number of Negative. Defaults to 0.

		Returns:
			float: FPR
		"""
		assert(FP>=0 or N>0), ValueError(
			'At least the Negative must be non null and False Positive more than 0')
		return FP/N
	
	@metric_catcher
	def compute_FDR(self, FP=0, TP=0):
		"""
		This function computes the False Discovery Rate (FDR).

		Args:
			FP (int, optional): Number of False Positive. Defaults to 0.
			TP (int, optional): Number of False Positive. Defaults to 0.

		Returns:
			float: FDR
		"""
		assert((FP !=0 and TP != 0) or FP >= 0 or TP >= 0), ValueError(
		"At least False Potive or True Positive must be non null and positive")
		return FP / (FP+TP)
	
	@metric_catcher
	def compute_FOR(self,FN=0, TN=0):
		"""
		This function computes the False Ommission Rate (FDR).

		Args:
			FN (int, optional): Number of False Negative. Defaults to 0.
			TN (int, optional): Number of True Negative. Defaults to 0.

		Returns:
			float: FOR
		"""
		assert((FN !=0 and TN != 0) or FN >= 0 or TN >= 0), ValueError(
		"At least False Negative or True Negative must be non null and positive")
		return FN / (FN+TN)
	
	@metric_catcher
	def compute_LRp(self, TPR:float=None, FPR:float=None):
		"""
		This function computes the Positive Likelihood Ration (LR+).

		Args:
			TPR (int, optional): True Positiv Rate. Defaults to 0.
			FPR (int, optional): False Positive Rate. Defaults to 0.

		Returns:
			float: LR+
		"""
		if TPR and FPR :
			assert(TPR>=0 or FPR>0), ValueError(
				'At least the True Positive Rate must be non null and False Psotive Rate more than 0')
		else :
			TPR = self.compute_TPR(TP=self.metrics['TP'], 
						  FN=self.metrics['FN'])
			FPR = self.compute_FPR(FP=self.metrcis['FP'],
						  N=self.metrics['N'])
		return TPR/FPR

	@metric_catcher
	def compute_LRn(self, FNR:float=None, TNR:float=None):
		"""
		This function computes the Negative Likelihood Ration (LR-).

		Args:
			FNR (float, optional): False Negative Rate. Defaults to None.
			TNR (float, optional): True Negative Rate. Defaults to None.

		Returns:
			float: LR-
		"""
		if FNR and TNR :
			assert(FNR>=0 or TNR>0), ValueError(
				'At least the False Negative Rate must be non null and True Negative Rate more than 0')
		else :
			FNR = self.compute_FNR(FN=self.metrics['FN'], 
						  P=self.metrics['P'])
			TNR = self.compute_TNR(TN=self.metrics['TN'], 
						  N=self.metrics['N'])
		return FNR/TNR
	
	@metric_catcher
	def compute_PT(self, FPR=None, TPR=None):
		"""
		This function computes the Prevalence Threshold.

		Args:
			FPR (float, optional): False Positive Rate. Defaults to None.
			TPR (float, optional): True Positive Rate. Defaults to None.

		Returns:
			float: PT
		"""
		if FPR and TPR :
			assert((FPR !=0 and TPR != 0) or FPR >= 0 or TPR >= 0), ValueError(
				'At least the False Negative Rate must be non null and True Negative Rate more than 0')
		else :
			FPR = self.compute_FPR(FP=self.metrics['FP'],
						  N=self.metrics['N'])
			TPR = self.compute_TPR(TP=self.metrics['TP'],
						  FN=self.metrics['FN'])
		return np.sqrt(FPR) / (np.sqrt(TPR) + np.sqrt(FPR))
	
	@metric_catcher
	def compute_TS(self, TP=0, FN=0, FP=0):
		"""
		This function computes the Threat Score (TS) or succes index (CSI).

		Args:
			TP (int, optional): Number of True Positive. Defaults to 0.
			FN (int, optional): Number of False Negative. Defaults to 0.
			FP (int, optional): Number of False Positive. Defaults to 0.

		Returns:
			float: TS
		"""
		assert((TP!=0 and FN !=0 and FP!=0) or TP>=0 or FN>=0 or FP>=0), ValueError(
			'At least not all null at the same time and must be positive')
		return TP / (TP+FN+FP)
		
	@metric_catcher
	def compute_prevalence(self, P=0, N=0):
		"""
		This function computes the Prevalence.

		Args:
			P (int, optional): Number of Positive. Defaults to 0.
			N (int, optional): Number of Negative. Defaults to 0.

		Returns:
			float: TS
		"""
		assert((P!=0 and N !=0 ) or P>=0 or N>=0 ), ValueError(
			'At least not all null at the same time and must be positive')
		return P / (P+N)
	
	@metric_catcher
	def compute_ACC(self, TP=0, TN=0, P=0, N=0):
		"""
		This function computes the ACCuracy.

		Args:
			TP (int, optional): Number of True Positive. Defaults to 0.
			TN (int, optional): Number of True Negative. Defaults to 0.
			P (int, optional): Number of Positive. Defaults to 0.
			N (int, optional): Number of Negative. Defaults to 0.

		Returns:
			float: Accuracy
		"""
		assert((P!=0 and N !=0 ) or P>=0 or N>=0 or TP>=0 or TN>=0 ), ValueError(
			'At least not all null at the same time and must be positive')
		return (TP + TN) / (P + N)
	
	@metric_catcher
	def compute_BA(self, TPR=None, TNR=None):
		"""This function computes the Balanced Accuracy.

		Args:
			TPR (float, optional): True Psotive Rate. Defaults to None.
			TNR (float, optional): True Negative Rate. Defaults to None.

		Returns:
			float: Balanced Accuracy
		"""
		if not(TPR and TNR):
			TPR = self.compute_TPR(TP=self.metrics['TP'],
						  FN=self.metrics['FN'])
			TNR = self.compute_TNR(TN=self.metrics['TN'],
						  N=self.metrics['N'])
		return (TPR + TNR) / 2
	
	@metric_catcher
	def compute_F1_score(self, PPV = None, TPR=None):
		"""This function computes the F1 score or harmonic mean of precision and sensitivity.

		Args:
			PPV (float, optional): Positive Predictive Value. Defaults to None.
			TPR (float, optional): True Positive Rate. Defaults to None.

		Returns:
			float: F1 score
		"""
		if not(PPV and TPR):
			PPV =self.compute_PPV(TP=self.metrics['TP'],
						 FP=self.metrics['FP'])
			TPR = self.compute_TPR(TP=self.metrics['TP'],
						  FN=self.metrics['FN'])
		return 2 * PPV*TPR / (PPV + TPR)



class SegmentationMetrics2D(Metrics):
	def __init__(self, pix_threshold = 0, feats_threshold=0, multiple_masks = False) -> None :
		"""This class computes the metric for the segmentation task for 2D images

		Args:
			threshold (int, optional): threshold needed to evaluate value of predicted. Defaults to 0.
			multiple_masks (bool, optional): in case there are multiple masks generated. Defaults to False.
		"""
		super().__init__()
		self.pix_threshold = pix_threshold #TODO look the value to be given (ask Alexandre ?)
		self.feats_threshold = feats_threshold # TODO check if needs for threshold in picels or feature map
		self.all_masks = {}
		self.all_AUC = {}

	def save(self, id='', masks = [], TP=0, FP=0, FN=0, AP=0, AUC=0) -> None :
		self.all_masks[id] = masks
		self.all_AUC[id] = AUC

	def compute_AUC(self, predicted = None, target = None) -> list[float]:
		"""		
		Compute the Area Under Curve, comparing the prediction and the target
		Get the number of pixels normalized according to the size of the image

		Args:
			predicted (_type_, optional): _description_. Defaults to None.
			target (_type_, optional): _description_. Defaults to None.

		Returns:
			_type_: _description_
		"""
		pix_predicted = 0
		pix_AUC = 0
		pix_ground_truth = 0
		pix_mask_value = 1
		# handles errors
		assert(not (predicted is None)), ValueError("Need to give a predicted input")
		assert(not (target is None)), ValueError("Need to give a target input")
		# checks the size of the inputs
		if not(len(predicted[0]) == len(target[0]) and len(predicted[0][0]) == len(target[0][0])):
			# TODO transform the sizes to fit
			predicted, target = self.haromnizing_size(predicted, target)
		# parses through the pictures to compare pixels
		dim_x, dim_y = len(target[0]), len(target[0][0])
		for pix_x in range(dim_x):
			for pix_y in range(dim_y):
				# TODO check if needs to be compared to target or to a mask value
				# and if inly the number of pixels is enough
				if predicted[pix_x][pix_y] == target[pix_x][pix_y] :
					pix_AUC += 1
				if predicted[pix_x][pix_y] == pix_mask_value:
					pix_predicted += 1
				if target[pix_x][pix_y] == pix_mask_value:
					pix_ground_truth += 1
		#TODO normalize the output according to the size of the images ?
		surface = dim_x * dim_y
		return pix_AUC/surface, pix_predicted/surface, pix_ground_truth/surface
		
	def haromnizing_size(self, predicted, target):
		NotImplemented
		return predicted_resized, target_resized

	def positive_sample_matching_pix(self):
		#TODO based on Fast and accurate cable detection using CNN
		# or compare to feature map
		NotImplemented
	

	def prediction_correct_pix(self, AUC=0, predicted=0, target=0, method = ''):
		"""Tells if the prediction based on the mask corresponds to target

		Args:
			AUC (_type_): _description_
			predicted (_type_): _description_
			target (_type_): _description_
			method (str, optional): _description_. Defaults to ''.

		Returns:
			_type_: _description_
		"""
		assert(not(AUC==0 and predicted==0 and target==0)), ValueError(
			'All the input must be non null')
		assert (method in EVAL_METHOD_PIX), ValueError(f'the evaluation method is not in {METHOD}')
		if method == 'AUC':
			to_compare = AUC 
		elif method == 'diff':
			to_compare = predicted
		rest = (target - to_compare)**2
		return rest <= self.pix_threshold

	def evaluate_model(self):
		#TODO
		NotImplemented
		

