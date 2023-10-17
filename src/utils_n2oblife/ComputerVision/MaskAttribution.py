from .ImageDataset import CurrentImage, CurrentDataset 

class MaskFiller(CurrentDataset): 
    def __init__(self, path:str=None, mask_value:int = 1) -> None:
        super.__init__(path)
        self.in_box = False
        self.mask_value = mask_value
    
    def _is_in_box(self) -> bool:
        return self.in_box
    
    def _is_border(self, pixel_value:int) -> bool:
        if pixel_value == self.mask_value:
            if self.in_box:
                self.in_box = False
            else :
                self.in_box = True
    
    def fill_image(self) -> None:
        if self.in_box:
            self.currentImage.pixel_data = self.mask_value
