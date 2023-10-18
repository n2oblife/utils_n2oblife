import os
from datasets import load_dataset
from PIL import Image
from matplotlib import image, pyplot
import numpy as np

class CurrentImage:
    def __init__(self, id=0, path:str=None, format='PNG') -> None:
        self.path = path
        self.image = Image()
        self.data = np.array()
        self.format = format #TODO handle other modes than RGB
        self.id = id
        self.dim_x = 0
        self.dim_y = 0
        self.pix_x = 0
        self.pix_y = 0
        self.pixel_data = np.array()

    def load_pixel(self, x:int, y:int)-> None:
        assert(x< self.dim_x and y<self.dim_y), ValueError(
            f"The pixel need to be in this {self.dim_x}x{self.dim_y} image")
        self.pix_x = x
        self.pix_y = y
        self.pixel_data = self.data[self.pix_y, self.pix_x]
    
    def get_pixel(self, x:int=None, y:int=None) -> tuple[int]:
        if x and y :
            self.load_pixel(x,y)
            return self.pixel_data
        elif self.pix_x and self.pix_y:
            return self.pixel_data
        else :
            raise ValueError("The pixel position needs to be initiated")

    def load_path(self, path:str=None)-> None:
        assert(path), ValueError("The path must be non null")
        self.path = path

    def load_current_image(self, id:int, path:str=None)->None:
        if path:
            self.path = path
        if self.path:
            self.image = Image.open(self.path)
        else :
            raise ValueError("The path needs to be instanciated")
        self.id = id
        self.dim_x, self.dim_y = self.image.size
        self.data = image.imread(self.path)
        self.format = self.image.format
    
    def plot_image(self):
        self.image.show()
    
    def pix_is_null(self)-> bool:
        if self.pixel_data:
            return not(self.pixel_data.any())
        else :
            return not(self.get_pixel(self.pix_x, self.pix_y).any())
        
    def is_pix_mask(self, value=1)->bool:
        """Checks if the pixel is composed of the same value

        Args:
            value (int, optional): Value to check. Defaults to 1.

        Returns:
            bool: _description_
        """
        value_array = np.full(self.pixel_data, value)
        value_set = {value}
        return  (set(self.pixel_data) & set(value_array)) == value_set
    
    def loop_pixels(self):
        """Iterates over the pixels of the image

        Yields:
            tuple: the pixel data and its (x,y) position
        """
        #TODO why not add some parameters
        for y, row in enumerate(self.data):
            for x, pixel in enumerate(row):
                # are the pixels and pixel_data really useful ?
                # self.pix_x, self.pix_y = x, y
                # self.pixel_data = self.data[y,x]
                yield pixel, x, y


class CurrentDataset:
    def __init__(self, path:str=None) -> None:
        self.path = path 
        self.all_images = []
        self.currentImage = CurrentImage(path=path)

    def load_dataset(self, path:str=None) -> None:
        # TODO handle the labels dataset
        assert(self.path and path), ValueError("The path to the dataset is not initiated")
        if path:
            self.change_path(path)
        self.all_images = os.listdir(self.path)

    def _get_path(self) -> str:
        return self.path

    def load_image(self, id:int=None, path:str=None):
        if self.all_images == [] or path:
            self.load_dataset(path=path)
        if id:
            self.currentImage.id = id
        self.currentImage.load_path(path=
                                    self.path+'/'+self.all_images[
                                        self.currentImage.id])
        self.currentImage.load_current_image(id=
                                             self.currentImage.id)
    
    def plot_image(self, id:int=None):
        if id :
            self.currentImage.load_current_image(id=id)
        self.currentImage.plot_image()
    
    def loop_all_images(self, path:str=None):
        """Itertaes over all the images of the current dataset or in the path

        Args:
            path (str, optional): Path from which loading the dataset. Defaults to None.

        Yields:
            CurrentImage: Custom class of image
        """
        for id,_ in enumerate(self.all_images):
            self.load_image(id=id, path=path)
            yield self.currentImage
