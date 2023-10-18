from .ImageDataset import CurrentImage, CurrentDataset 
import numpy as np
from PIL import Image

class EdgeBucket:
    '''class to implement the scanline filling algorithm'''
    def __init__(self) -> None:
        self.ymax:int
        self.xofymin:int
        self.slopeinverse:int

class EdgeTuple:
    '''class to implement the scanline filling algorithm'''
    def __init__(self) -> None:
        self.number_of_edge_buckets = 0
        self.ede_list:list[EdgeBucket]

class MaskFiller(CurrentDataset): 
    def __init__(self, path:str=None, mask_value:int = 1) -> None:
        super.__init__(path)
        self.in_box = False
        self.mask_value = mask_value
        self.mask_array = np.array([])
        # self.number_of_box_line_cache = 0 # might not be useful
        # self.current_line_cache = []
        self.edge_table:list[EdgeTuple] = []
        self.active_list:EdgeTuple
    
    def check_current_image(self)->None:
        '''Checks if the current image is loaded.
        Loads it otherwise'''
        if not(self.current_image):
            self.load_image(self.path)

    def check_pixels_mask(self,current_pix:bool, cache_pixel:bool):
        #revize algo
        if cache_pixel and current_pix:
            self.in_box = True
        elif cache_pixel ^ current_pix: #xor
            self.in_box = not self.in_box
        else :
            self.in_box = False

    def check_in_box(self, current_pix:bool, cache_pixel:bool, beg_of_line=False, first_line=False):
        #TODO WARNING if problem of completion look at this function
        # could have in cache the position of the last pixel found (for each segment) to have a guess on the beggining of the new line otherwise might not work
        # this function needs to be more complex, don't mess with too much complexity !
        # first approximation : the lines end on the whole image
        #we can also assume that the edge is useless, what is worth is the center
        if first_line :
            self.check_pixels_mask(current_pix=current_pix,
                                   cache_pixel=cache_pixel)
    
    def boundary_fill(self, x:int, y:int)-> None:
        """The boundary fill algorithm adapted from : https://www.geeksforgeeks.org/boundary-fill-algorithm/
        
        This algorithm fills a given polygon with the mask array from any center point.

        Args:
            x (int): _description_
            y (int): _description_
        """
        if not self.current_image.is_pix_mask(x,y):
            self.fill_pixel(x,y)
            for val_x in (-1, 0, 1):
                for val_y in (-1, 0, 1):
                    self.boundary_fill(x=x+val_x, y=y+val_y)
    
    def scan_line_fill(self, x:int, y:int)->None:
        NotImplemented

            
    def is_in_box(self):
        return self.in_box
    
    # def cross_border(self, pixel_value:int) -> bool:
    #     self.check_current_image()
    #     if self.current_image:
    #         if self.in_box:
    #             self.in_box = False
    #         else :
    #             self.in_box = True

    def fill_image(self, output_path='.') -> None:
        self.check_current_image()
        cache_pixel_data, cache_pixel_bool= np.array([0,0,0]), False
        pixel_bool = self.current_image.is_pix_mask()
        for pixel,x,y,eol in self.current_image.loop_pixels():
            #TODO let's see this relation and put some things in a function to do that a first time out of the loop
            self.check_in_box(current_pix=pixel_bool, 
                              cache_pixel=cache_pixel_bool,
                              beg_of_line= (x==0),
                              first_line= (y==0))
            
            if self.in_box:
                self.fill_pixel()
            pixel_bool, cache_pixel_bool = self.current_image.is_pix_mask(), pixel_bool
            cache_pixel_data = pixel
        self.save_image(output_path)

    def fill_pixel(self, x:int, y:int, mask_value:int=None):
        if 0<=x<self.current_image.dim_x and 0<=y<self.current_image.dim_y:
            if mask_value:
                self.mask_value = mask_value
                self.mask_array = np.full(self.current_image.pixel_data.shape, 
                                        self.mask_value)
            elif not self.mask_array:
                self.mask_array = np.full(self.current_image.pixel_data.shape, 
                                        self.mask_value)
            self.current_image.pixel_data = self.mask_array
        # TODO finish !!!

    def save_image(self, output_path='.'):
        self.current_image.image = Image.fromarray(self.current_image.data)
        self.current_image.image.save(output_path, 
                                     format=self.current_image.format)
    

class MaskAttribution:
    def __init__(self) -> None:
        pass