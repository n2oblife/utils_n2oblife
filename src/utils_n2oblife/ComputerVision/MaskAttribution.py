from .ImageDataset import CurrentImage, CurrentDataset 
import numpy as np
from PIL import Image

class MaskFiller(CurrentDataset): 
    def __init__(self, path:str=None, mask_value:int = 1) -> None:
        super.__init__(path)
        self.in_box = False
        self.mask_value = mask_value
        self.mask_array = np.array([])
    
    def check_current_image(self)->None:
        '''Checks if the current image is loaded.
        Loads it otherwise'''
        if not(self.currentImage):
            self.load_image(self.path)

    def check_in_box(self, current_pix:bool, cache_pixel:bool):
        if cache_pixel and current_pix:
            self.in_box = True
        elif cache_pixel ^ current_pix: #xor
            self.in_box = not self.in_box
        else :
            self.in_box = False

    def is_in_box(self):
        return self.in_box
    
    # def cross_border(self, pixel_value:int) -> bool:
    #     self.check_current_image()
    #     if self.currentImage:
    #         if self.in_box:
    #             self.in_box = False
    #         else :
    #             self.in_box = True

    def fill_image(self, output_path='.') -> None:
        self.check_current_image()
        cache_pixel_data = np.array([0,0,0])
        for y,row in enumerate(self.currentImage.pixel_data[]):
            for x,pixel in enumerate(row):
                #TODO let's see this relation and put some things in a function to do that a first time out of the loop
                pixel_bool, cache_pixel_bool = self.check_pixel()
                self.check_in_box()
                if self.in_box:
                    self.fill_pixel()
                cache_pixel_data = pixel
        self.save_image(output_path)

    def fill_pixel(self, x, y, mask_value:int=None):
        if mask_value:
            self.mask_value = mask_value
            self.mask_array = np.full(self.currentImage.pixel_data.shape, 
                                    self.mask_value)
        elif not self.mask_array:
            self.mask_array = np.full(self.currentImage.pixel_data.shape, 
                                    self.mask_value)
        self.currentImage.pixel_data = self.mask_array
        # TODO finish !!!

    def save_image(self, output_path='.'):
        self.currentImage.image = Image.fromarray(self.currentImage.data)
        self.currentImage.image.save(output_path, 
                                     format=self.currentImage.format)
    
    def correct_filling(self, output_path=''):
        NotImplemented

