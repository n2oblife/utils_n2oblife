from typing import Any
import torchvision.transforms as transforms
from .Metrics import *

class Evaluate:
    def __init__(self, model_path:str=None, model=None, 
                 model_name:str=None, stats:ClassicMetrics=None, 
                 dataset_path:str=None, batch_size = 32, ) -> None:
        self.model_path = model_path
        self.model = model
        self.dataset_path = dataset_path
        self.dataset = None
        self.stats = stats
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (), ()), # add normalization adapted for the model, not PIL
        ])
    
    def load_evaluation(self):
        pass

    def load_model(self):
        pass

    def load_metrics(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

class Segmentation2DEvaluate(Evaluate):
    # def init

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

# class ClassifEvaluate(Evaluate):
