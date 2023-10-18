from .ImageDataset import ImageDataset


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

class MaskFiller(ImageDataset):
    """An ImageDataset like class to file the edges into segmentation masks.

    Args:
        path (str): Path to the dataset folder.
        mask_pixel(tuple[int,int,int]): The mask pixel value (white)
    """
    def __init__(self, path: str = None, mask_pixel=(255,255,255)) -> None:
        super().__init__(path)
        # self.path = path
        # self.all_images = []
        # self.current_image = CurrentImage()
        self.mask_pixel = mask_pixel
        ## load only for the scanline function
        self.edge_table:list[EdgeTuple] = []
        self.active_list:EdgeTuple

    def boundary_fill(self, x:int, y:int)->None:
        """The boundary fill algorithm adapted from : https://www.geeksforgeeks.org/boundary-fill-algorithm/
        
        This algorithm fills a given polygon with the mask array from any center point.

        Args:
            x (int): Position of the starting pixel to fill on x-axis.
            y (int): Position of the starting pixel to fill on y-axis.
        """
        if 0<=x<self.current_image.dim_x and 0<=y<self.current_image.dim_y:
            if not self.current_image.is_pix_mask(x,y,self.mask_pixel):
                self.current_image.fill_pixel(x,y,self.mask_pixel)
                for val_y in (-1, 0, 1):
                    for val_x in (-1, 0, 1):
                        self.boundary_fill(x+val_x, y+val_y)
        
    def scan_line_fill(self, x:int, y:int):
        """The scan line fill algorithm adapted from : file:///C:/Users/ZaccarieKanit/Zotero/storage/G6YIKV83/scan-line-polygon-filling-using-opengl-c.html
        
        This algorithm fills a given polygon with the mask array from any center point.

        Args:
            x (int): Position of the starting pixel to fill on x-axis.
            y (int): Position of the starting pixel to fill on y-axis.
        """
        NotImplemented
        #TODO