import os
import torch
import numpy as np
from numpy import random

class Device():
    def __init__(self) -> None:
        self._gpu = torch.cuda.is_available()
    
    def load_seed(self):
        # set seed for reproductibility
        print('Setting up seed...')
        seed  = 123456789
        try :
            os.environ['PYTHONHASHSEED'] = str(seed)
        except :
            pass
        try :
            random.seed(seed)
        except :
            pass
        try :
            np.random.seed(seed)
        except :
            pass
        torch.manual_seed(seed)
        if self._gpu :
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # empty cache
            torch.cuda.empty_cache()

class Device_CEA(Device):
    def __init__(self) -> None:
        """
        Class used in the lima project : https://github.com/aymara/lima.git
        Almost a config which changes depending on the device used
        """
        super().__init__()
        base_dir = '/home/users/zkanit/'
        self._base_dir = base_dir
        self._save_dir = base_dir+''
        self._data = base_dir+''