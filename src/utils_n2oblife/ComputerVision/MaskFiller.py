from .ImageDataset import ImageDataset
from tqdm import tqdm
# from utils_n2oblife.InterractionHandling.ScriptsUtils import restart_line
# import sys
# sys.setrecursionlimit(10000000)

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
    def __init__(self, path:str = None, save_file='.', mask_pixel=(255,255,255)) -> None:
        super().__init__(path)
        # self.path = path
        # self.all_images = []
        # self.current_image = CurrentImage()
        self.mask_pixel = mask_pixel
        ## load only for the scanline function
        self.edge_table:list[EdgeTuple] = []
        self.active_list:EdgeTuple
        self.filling_function = self.boundary_fill
        self.save_file = save_file

    def boundary_fill(self, x:int, y:int)->None:
        """The boundary fill algorithm adapted from : https://www.geeksforgeeks.org/boundary-fill-algorithm/
        
        This algorithm fills a given polygon with the mask array from any center point. Too much recursions.

        Args:
            x (int): Position of the starting pixel to fill on x-axis.
            y (int): Position of the starting pixel to fill on y-axis.
        """
        if 0<x<self.current_image.dim_x and 0<y<self.current_image.dim_y:
            if not self.current_image.is_pix_mask(x,y,self.mask_pixel):
                self.current_image.fill_pixel(x,y,self.mask_pixel)
                for val_y in (0, 1):
                    for val_x in (0, 1):
                        if not(val_y==0 and val_x==0):
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
    
    def fill_from_line(self, edges_list:list):
        """You can change the filling function with the parameters
        """
        n_elt = len(edges_list)
        for i in range(0, n_elt, 2):
            edge = edges_list[i]
            self.filling_function(x=edge[0]+1, y=edge[1])
            self.current_image.fill_pixel(edge[0],edge[1], (255,0,0))
            self.current_image.fill_pixel(edges_list[i+1][0],edges_list[i+1][1], (255,0,0))

    def fill_dataset(self):
        edges_coordinates = []
        n_images = len(self.all_images)
        for img in self.loop_all_images():
            progress = tqdm(total=n_images+1, 
                            desc=f"Processing image {img.id+1}/{n_images}",
                            ncols=75)
            for _,x,y in img.loop_pixels():
                progress.update(1)
                if x==0:
                    # reset the edges at the begining each line
                    edges_coordinates = []
                if x==img.dim_x-1:
                    length = len(edges_coordinates)
                    if length % 2 == 0 and length!=0:
                        self.fill_from_line(edges_coordinates)
                        # to debug the border
                        self.plot_current_image()
                        breakpoint()
                else :
                    if img.is_border(x,y):
                        #img.fill_pixel(x,y, (255,0,0))
                        edges_coordinates.append((x,y))
            progress.close()
            self.save_current_image(self.save_file)