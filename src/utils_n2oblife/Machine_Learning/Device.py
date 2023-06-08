class Device_CEA():
    def __init__(self) -> None:
        import torch

        if torch.cuda.is_available():
            # FactotyAI
            base_dir = '/home/users/zkanit/'
        else :
            # PC
            base_dir = '/home/zk274707/Projet/proto/'
        
        self._base_dir = base_dir
        self._save_dir = base_dir+''
        self._data = base_dir+''
        self._gpu = torch.cuda.is_available()
        self._