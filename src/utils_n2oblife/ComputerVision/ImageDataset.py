import os
from datasets import load_dataset
from PIL import Image
from matplotlib import image, pyplot
import numpy as np

class CurrentImage:
    def __init__(self, id = 0, path:str=None) -> None:
        self.path = path
        self.image = Image()
        self.data = np.array()
        self.mode = ''
        self.id = id
        self.dim_x = 0
        self.dim_y = 0
        self.pix_x = 0
        self.pix_y = 0
        self.pixel_data = None

    def load_pixel(self, x, y)-> None:
        assert(x< self.dim_x and y<self.dim_y), ValueError(
            f"The pixel to get need to be in this {self.dim_x}x{self.dim_y} image")
        self.pix_x = x
        self.pix_y = y
        self.pixel_data = self.data[self.pix_x,self.pix_y]
    
    def get_pixel(self, x, y) -> tuple[int]:
        return self.pixel_data

    def load_path(self, path:str=None)-> None:
        assert(path), ValueError("The path must be non null")
        self.path = path

    def load_current_image(self, id:int):
        self.image = Image.open(self.path)
        self.id = id
        self.dim_x, self.dim_y = self.image.size
        self.data = image.imread(self.path)
    
    def plot_image(self):
        self.image.show()

class CurrentDataset:
    def __init__(self, path:str=None) -> None:
        self.path = path 
        self.all_images = []
        self.currentImage = CurrentImage(path=self.path)

    def _load_dataset(self, path:str=None) -> None:
        # TODO handle the labels dataset
        assert(self.path and path), ValueError("The path to the dataset is not initiated")
        if path:
            self.change_path(path)
        self.all_images = os.listdir(self.path)

    def _get_path(self) -> str:
        return self.path

    def load_image(self, id=0, path:str=None):
        if self.all_images == []:
            self.load_dataset(path=path)
        self.id = id
        self.currentImage.load_path(path=self.path+'/'+self.all_images[id])
        self.currentImage.load_current_image(id=id)
    
    def plot_image(self, id:int=None):
        if id :
            self.currentImage.load_current_image(id=id)
        self.currentImage.plot_image()
