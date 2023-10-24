from .ImageDataset import ImageDataset
from tqdm import tqdm
import numpy as np
from PIL import ImageDraw
import sys
sys.setrecursionlimit(10000)

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
        self.edges_coordinates = []
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
        if 0<=x<self.current_image.dim_x and 0<=y<self.current_image.dim_y:
            if not (self.current_image.is_pix_mask(x,y,(255,0,0)) or
                    self.current_image.is_pix_mask(x,y,(255,255,255))) :
                # self.current_image.fill_pixel(x,y,(0,255,0))
                #self.plot_current_image()
                #breakpoint()
                self.current_image.fill_pixel(x,y,self.mask_pixel)
                for val_y in (0, 1):
                    for val_x in (-1, 0, 1):
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

    def find_inside(self, edges_list:list, far_away=10):
        rand_id = np.random.randint(0,len(edges_list)) 
        rand_x, rand_y = edges_list[rand_id]
        final_x, final_y = rand_x, rand_y
        for i in range(1,far_away):
            if (rand_x+i, rand_y) in edges_list:
                final_x = rand_x+i//2
            elif (rand_x-i, rand_y) in edges_list:
                final_x = rand_x+i//2
        for j in range(1, far_away):
            if (final_x, rand_y+j) in edges_list:
                final_y = rand_y+j//2
            elif (final_x, rand_y-j) in edges_list:
                final_y = rand_y+j//2
        return final_x,final_y
        
    def fill_from_edges(self, edges_list:list):
        x,y = self.find_inside(edges_list, far_away=6)
        self.filling_function(x,y)

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

    def find_border(self, x:int, y:int):
        self.current_image.define_border(x,y)
        # self.fill_from_edges(
        #     self.current_image.current_border)
        self.edges_coordinates.append(
            self.current_image.current_border)
        
    def fill_dataset_PIL(self):
        n_images = len(self.all_images)
        pixel_in_edges = False
        for img in self.loop_all_images():
            progress = tqdm(total=n_images+1, 
                            desc=f"Processing image {img.id+1}/{n_images}",
                            ncols=75)
            self.edges_coordinates = []
            pixel_in_edges = False
            for _,x,y in img.loop_pixels():
                progress.update(1)
                if self.current_image.is_pix_mask(x,y):
                    if len(self.edges_coordinates) == 0:
                        self.find_border(x,y)
                    else :
                        for edges in self.edges_coordinates:
                            if not (x,y) in edges:
                                pixel_in_edges = True
                    if pixel_in_edges:
                        self.find_border(x,y)
                        # need to sort clockwise or coutner clockwise
                        # https://www.baeldung.com/cs/sort-points-clockwise
                    tmp_draw = ImageDraw.Draw(
                        self.current_image.image
                        )
                    tmp_draw.polygon(
                        xy=self.current_image.current_border,
                        fill ="blue", outline ="blue"
                        )
                    self.current_image.current_border = []
                    #self.filling_function(x+1,y+1)
                    self.plot_current_image()
                    breakpoint()
                # if len(self.edges_coordinates) == 4:
                #     break
            # self.plot_current_image()
            progress.close()
            self.save_current_image(self.save_file)
            print(len(self.edges_coordinates))


    def get_center(self, edges_list:list):
        """
        Returns:
            tuple[int, int]: coodinates of the center of polygon
        """
        sx = sy = sL = 0
        for i in range(len(edges_list)):
            x0, y0 = edges_list[i - 1]     
            x1, y1 = edges_list[i]
            L = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
            sx += (x0 + x1)/2 * L
            sy += (y0 + y1)/2 * L
            sL += L
        # return x_center, y_center
        return sx//sL, sy//sL