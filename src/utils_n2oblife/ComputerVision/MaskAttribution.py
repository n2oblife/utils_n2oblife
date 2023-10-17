from datasets import load_dataset
from PIL import 

class CurrentImage:
    def __init__(self, id = 0) -> None:
        self.image = None
        self.id = id
        self.dim_x = 0
        self.dim_y = 0
        self.pix_x = 0
        self.pix_y = 0
        self.pix_value_list = (0, 0, 0)
        self.pix_value = 0

    def load_pixel(self, x, y)-> None:
        assert(x< self.dim_x and y<self.dim_y), ValueError(
            f"The pixel to get need to be in this {self.dim_x}x{self.dim_y} image")
        self.pix_x = x
        self.pix_y = y
        self.pix_value_list = self.image.getpixel((x,y))
    
    def get_pixel(self, x, y) -> tuple[int]:
        self.load_pixel(x,y)
        return self.pix_value_list

    def load(self, image=None, id=0)-> None:
        self.image = image
        self.id = id
        self.dim_x, self.dim_y = image.size
        self.pix_value_list = self.image.getpixel((0,0))

class MaskFiller:
    def __init__(self, path=None, mask_value:int = 1) -> None:
        self.in_box = False
        self.path = path
        self.mask_value = mask_value
        self.dataset = None
        self.image = CurrentImage()
    
    def _is_in_box(self) -> bool:
        return self.in_box
    
    def change_path(self, given_path) -> None:
        self.path = given_path

    def _get_path(self) -> str:
        return self.path
    
    def _load_dataset(self, path = None) -> None:
        # TODO handle the labels dataset
        assert(self.path and path), ValueError("The path to the dataset is not initiated")
        if path:
            self.change_path(path)
        self.dataset = load_dataset(path= self.path)
    
    def _load_image(self, id=0)-> None:
        self.image.load(
            image=self.dataset['train'][id]['image'],
            id=id
        )
    
    def _is_border(self, pixel_value:int) -> bool:
        if pixel_value == self.mask_value:
            if self.in_box:
                self.in_box = False
            else :
                self.in_box = True
    
    def fill_image(self) -> None:
        if self.in_box:




