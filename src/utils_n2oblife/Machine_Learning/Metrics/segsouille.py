from .ClassicMetrics import Metrics

EVAL_METHOD = ['AUC', 'IoU' ]

# already implemented on tensorflow
class SegmentationMetrics2D(Metrics):
	def __init__(self, threshold = 0., multiple_masks = False) -> None :
		"""This class computes the metric for the segmentation task for 2D images

		Args:
			threshold (float, optional): threshold needed to evaluate value of predicted. Defaults to 0.
			multiple_masks (bool, optional): in case there are multiple masks generated. Defaults to False.
		"""
		super().__init__(self)
		self.threshold = threshold #TODO look the value to be given (ask Alexandre ?)
		self.all_predicted_masks = {}
		self.qualitative_eval = {} # comparing the mask to threshold 
		self.all_eval = {}

    # def load(self, predicted, target):
    #     self.

	def save(self, id='', masks = [], TP=0, FP=0, FN=0, AP=0, AUC=0) -> None :
		NotImplemented
	
    def compute_AUC(self, predicted = None, target = None) -> list[float]:
		# equivalent to IoU metrics (let's see pytorch)
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
		#return predicted_resized, target_resized

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
		assert (method in EVAL_METHOD), ValueError(f'the evaluation method is not in {EVAL_METHOD}')
		if method == 'AUC':
			to_compare = AUC 
		elif method == 'diff':
			to_compare = predicted
		rest = (target - to_compare)**2
		return rest <= self.pix_threshold

	def evaluate_model(self):
		#TODO
		NotImplemented