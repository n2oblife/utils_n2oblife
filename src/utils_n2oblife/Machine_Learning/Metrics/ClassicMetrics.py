#This class is an implementation of some metrics 
#some can be seen here : https://en.wikipedia.org/wiki/Sensitivity_and_specificity

import time
import functools
import numpy as np

METRICS = ['P','N','TP', 'TN', 'FP', 'FN', 
		   'TPR','FNR', 'TNR', 'FPR',
		   'PPV', 'FDR', 'NPV', 'FOR', 'LR+', 'LR-', 'PT', 'TS',
		   'prev', 'ACC', 'BA', 'F1']

def metric_catcher(f):
	@functools.wraps(f)
	def func(*args, **kwargs):
		try:
			return f(*args, **kwargs)
		except Exception as e:
			print(f'Need to initiate the metrics correctly and save them')
	return func

#TODO all values here are int but it should be any
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

