#This class is an implementation of some metrics 
#some can be seen here : https://en.wikipedia.org/wiki/Sensitivity_and_specificity

class Metrics():
	def __init__(self) -> None:
		"""a class template to be adapted for any kind of task (for now specific to ML)
		"""
		pass
	
	def compute_mean_distance(self, predicted = None, target = None, power = 1, n_elts = 2) -> float:
		"""This function compute the mean distance, 
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

	def compute_PPV(self, TP = 0, FP = 0) -> float:
		"""This function compute the precision or the positive prediction value

		Args:
			TP (int, optional): number of True Positive. Defaults to 0.
			FP (int, optional): number of False Positive. Defaults to 0.

		Returns:
			float: PPV
		"""
		assert((TP !=0 and FP != 0) or TP >= 0 or FP >= 0), ValueError(
		"At least True positive or False positive must be non null and positive")
		return TP / (TP + FP)

	def compute_TPR(self, TP = 0, FN = 0) -> float:
		"""This function compute the recall or the true positive rate

		Args:
			TP (int, optional): number of True Positive. Defaults to 0.
			FN (int, optional): number of False Negative. Defaults to 0.

		Returns:
			float: TPR
		"""
		assert((TP !=0 and FN != 0) or TP >= 0 or FN >= 0), ValueError(
		"At least True positive or False negative must be non null and positive")
		return TP / (TP + FN)

class SegmentationMetrics2D(Metrics):
	def __init__(self, threshold = 0,multiple_masks = False) -> None :
		"""This class computes the metric for the segmentation task for 2D images

		Args:
			threshold (int, optional): threshold needed to evaluate value of predicted. Defaults to 0.
			multiple_masks (bool, optional): in case there are multiple masks generated. Defaults to False.
		"""
		super().__init__()
		self.Ta = threshold
		self.all_masks = {}
		self.all_TP = {}
		self.all_FP = {}
		self.all_FN = {}
		self.all_AP = {}
		self.all_AUC = {}

	def save(self, id='', masks = [], TP=0, FP=0, FN=0, AP=0, AUC=0) -> None :
		self.all_masks[id] = masks
		self.all_TP[id] = TP
		self.all_FP[id] = FP
		self.all_FN[id] = FN
		self.all_AP[id] = AP
		self.all_AUC[id] = AUC

	def compute_AUC(self, predicted = None, target = None) -> None:
		'''
		Compute the Area Under Curve, comparing the prediction and the target with 
		precision andd according to a threshold
		'''
		pix_TP = 0
		pix_FP = 0
		pix_FN = 0
		pix_ground_truth = 0
		if not(len(predicted[0]) == len(target[0]) and len(predicted[0][0]) == len(target[0][0])):
			# transform the sizes to fit
			pass
		dim_x, dim_y = len(target[0]), len(target[0][0])
		for pix_x in range(dim_x):
			for pix_y in range(dim_y):
				if predicted[pix_x][pix_y] == target[pix_x][pix_y] :
					pix_AUC += 1

	def evaluate_result(self, predicted, target):
		NotImplementedError
		