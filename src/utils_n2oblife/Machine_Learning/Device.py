import os
import torch
import numpy as np
import random

class Device():
    def __init__(self, seed  = 123456789) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = seed
    
    def load_seed(self):
        # set seed for reproductibility
        try :
            import os
            os.environ['PYTHONHASHSEED'] = str(self.seed)
        except :
            pass
        try :
            import random
            random.seed(self.seed)
        except :
            pass
        try :
            import numpy as np
            np.random.seed(self.seed)
        except :
            pass
        try:
            import tensorflow as tf
            tf.set_random_seed(self.seed)
        except :
            pass
        try:
            import tensorflow as tf
            from keras import backend as K
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)
        except:
            pass
        try:
            import torch
            torch.manual_seed(self.seed)
            if self.gpu :
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                # empty cache
                torch.cuda.empty_cache()
        except:
            pass

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