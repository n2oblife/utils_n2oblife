import os
import torch
from tqdm import tqdm
from typing import Any
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 

from utils_n2oblife.Machine_Learning.Metrics.ClassicMetrics import *
from utils_n2oblife.Machine_Learning.Device import Device
from utils_n2oblife.ComputerVision.ImageDataset import ImageDatasetTorch

class Evaluate:
    def __init__(self, model_path:str=None, model=None, 
                 model_name:str=None, metrics:Metrics=None, 
                 dataset_path:str=None, label_path:str=None,
                 batch_size = 32, 
                 device:Device=None) -> None:
        self.model_path = model_path
        self.model = model
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.dataset = ImageDatasetTorch(path=dataset_path,
                                         label_path=label_path)
        self.metrics = metrics
        self.stats = {}
        self.device = device
        # self.transform = transforms.Compose([
        #     # transforms.Normalize(
        #     #     (), ()), # add normalization adapted for the model, not PIL
        # ])
    
    def load_evaluation(self, metrics:Metrics=None,
                        device:Device=None):
        if metrics:
            self.metrics = metrics
        if device :
            self.device = device

    def load_model_from_path(self, path:str):
        #TODO add ways of loading models
        self.model_path = path
        split_str = os.path.splitext(self.model_path)
        if (split_str[-1] == '.pt' or split_str[-1] == '.pth' 
            or split_str[-1] == '.pwf' or split_str[-1] == '.ptc'):
            temp = torch.load(self.model_path)
            if 'state_dict' in temp.keys():
                self.model = temp['state_dict']
            elif split_str[0].split('/')[-1][:3] == 'sam':
                from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
                # load the smallest model for speed
                sam = sam_model_registry["vit-b"](checkpoint=self.model_path)
                self.model = SamAutomaticMaskGenerator(sam)
                #TODO check
                self.mode.__call__ = self.model.generate
            else :
                self.model = temp
            # TODO add some elif to handle other cases
        elif split_str[-1] == '.engine':
            print('engine not built')
        # elif 'deploy.yaml' in os.listdir(self.model_path):
            #TODO write the deploy loading
        # elif split_str[0].split('/')[-1] == 'm3d_rpn_depth_aware_test':
        #     from importlib import import_module
        #     from easydict import EasyDict as edict
        #     from lib.imdb_util import *
            
        #     conf_path = split_str[0].split('/')[:-1]+'m3d_rpn_depth_aware_test_config.pkl'
        #     conf = edict(pickle_read(conf_path))
        #     conf.pretrained = None #check if needed

        #     self.model = import_module('models.' + conf.model).build(conf)
        #     load_weights(self.model, self.model_path, remove_module=True)
        try:
            self.model.eval()
        except: #TODO add exception to catch
            pass
        temp = split_str[0].split('/')
        self.model_name = temp[-1]
        

    def load_model(self, model_path:str=None, model=None, 
                   model_name:str=None):
        if model_path :
            self.load_model_from_path(model_path)
        if model :
            self.model = model
        if model_name :
            self.model_name = model_name
        #TODO finish loading models orrectly
        

    def load_metrics(self, metrics:Metrics=None):
        if metrics:
            self.metrics = metrics
            
    def prepare_evaluation(self):
        self.device.load_seed()
        try :
            self.model.eval()
        except :
            pass
        # TODO check the data gestion
        valid_model = self.model.to(self.device.device).eval()
        valid_dataset = self.dataset
        return valid_model, valid_dataset

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class Segmentation2DEvaluate(Evaluate):
    # def init

    def __call__(self, path_to_file:str=None, *args: Any, **kwds: Any) -> Any:
        valid_model, valid_dataset = self.prepare_evaluation()
        n_elt = len(self.dataset)
        progress = tqdm(total=n_elt, ncols=75, 
                        desc=f'Evaluating the model {self.model_name}')
        with torch.no_grad():
            for _,(images, labels) in valid_dataset.loop_all_dataset():
                progress.update(1)
                images.to(self.device)
                labels.to(self.device.device)

                # TODO check how to call a modelmight be problems
                predicted = valid_model(images)

                #TODO check if one or multiple masks


                stats

                #TODO report stats
        progress.close()
        return super().__call__(*args, **kwds)

# class ClassifEvaluate(Evaluate):
