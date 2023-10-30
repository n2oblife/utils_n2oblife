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
        ## other parameters in this class
        # self.path = path
        # self.all_images = []
        # self.current_image = CurrentImage()
        self.mask_pixel = mask_pixel
        self.edges_coordinates = []
        self.save_file = save_file
        ## load only for the scanline function
        # self.edge_table:list[EdgeTuple] = []
        # self.active_list:EdgeTuple

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

    def get_angle(self, center:tuple, point:tuple):
        x = point[0] - center[0]
        y = point[1] - center[1]
        angle = np.arctan2(y,x)
        if angle <= 0 :
            angle = 2*np.pi + angle
        return angle
    
    def get_distance(self, pt1:tuple, pt2:tuple):
        x = pt1[0] - pt2[0]
        y = pt1[1] - pt2[1]
        return np.sqrt(x*x + y*y)
    
    def merge(self, left:dict, right:dict)->dict:
        """Function used for timsort adapted for this situation
        """
        if len(left) == 0:
            return right
        if len(right) == 0:
            return left
        result = []
        index_left = index_right = 0

        while len(result) < len(left) + len(right):
            # we use a cursor on the arrays while comparing the elements to build new array 
            if left[index_left][1] <= right[index_right][1]:
                result.append(left[index_left])
                index_left += 1
            else:
                result.append(right[index_right])
                index_right += 1
            # at the end of the array just add the rest of the list
            if index_right == len(right):
                result += left[index_left:]
                break
            if index_left == len(left):
                result += right[index_right:]
                break
        return result

    def modified_insertion_sort(self, array:dict, left=0, right:int=None)->dict:
        """Used only for the timsort algorithm"""
        if right is None:
            right = len(array) - 1

        for i in range(left + 1, right + 1):
            key_item = array[i] 
            j = i - 1
            while j >= left and array[j][1] > key_item[1]: # comapring the angle
                array[j + 1] = array[j]
                j -= 1
            array[j + 1] = key_item
        return array

    def timsort(self, array:dict, min_run = 32)->list:
        """A O(n*log2(n)) algorithm to sort array, adapted for this work.

        Returns a list of the ids sorted by angle.
        """
        n = len(array)
        # Start by slicing and sorting small portions of the array
        for i in range(0, n, min_run):
            self.modified_insertion_sort(array, i, min((i + min_run - 1), n - 1))

        # Merge the sorted slices.
        # Start from `min_run`, doubling the size until surpassing array's length
        size = min_run
        while size < n:
            for start in range(0, n, size * 2):
                midpoint = start + size - 1
                end = min((start + size * 2 - 1), (n-1))
                merged_array = self.merge(
                    left=array[start:midpoint + 1],
                    right=array[midpoint + 1:end + 1])
                array[start:start + len(merged_array)] = merged_array
            size *= 2
        return [elt[0] for elt in array]

    def sort_clockwise(self, edges_list:list):
        """A O(n*log2(n)) sorting algorithm based on this article : https://www.baeldung.com/cs/sort-points-clockwise

        Args:
            edges_list (list): coordonates to sort clockwise

        Returns:
            list: a sorted array of the coordinates by angle
        """
        # point from which computing the angles with points
        center_pt = self.get_center(edges_list)
        # angles computed in radian counter-clockwise to a 0 radian
        all_angles = []
        for id, coordinates in enumerate(edges_list):
            angle = self.get_angle(center=center_pt,
                                   point=coordinates)
            distance = self.get_distance(center_pt, coordinates)
            all_angles.append((id, angle, distance))
        # we use the timsort because datas are all almost sorted sequencially
        sorted_ids = self.timsort(all_angles)
        sorted_list = [edges_list[id] for id in sorted_ids]
        return sorted_list

    def draw_mask(self, polygon:list=None, color='blue'):
        """Draws a mask based on the current border computed or a list of edges of a polygon.

        Args:
            polygon (list, optional): The very polygon. Defaults to None.
            color (tuple, optional): Color to fill with the polygon. Defaults to (255,255,255).
        """
        tmp_draw = ImageDraw.Draw(self.current_image.image)
        if polygon :
            if len(polygon)>1:
                ImageDraw.Draw(self.current_image.image).polygon(
                    polygon, outline=color, fill=color)
        else:
            if len(self.current_image.current_border)>1:
                ImageDraw.Draw(self.current_image.image).polygon(
                    self.current_border, outline=color, fill=color)

    def find_border(self, x:int, y:int):
        self.current_image.define_border(x,y)
        self.edges_coordinates.append(
            self.current_image.current_border)
        
    def fill_dataset(self):
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
                    # need to find if the center is on a border or inside
                    # to avoid the try 
                    # (usually inside but sometime in a border)
                    try :
                        sorted_border = self.sort_clockwise(
                            self.current_image.current_border
                            )
                    except ZeroDivisionError:
                        pass
                    self.draw_mask(polygon=sorted_border) 
                    # if len(sorted_border) > 1: 
                    #     tmp_draw = ImageDraw.Draw(
                    #         self.current_image.image
                    #         )
                    #     tmp_draw.polygon(
                    #         xy=sorted_border,
                    #         fill ="blue", outline ="blue"
                    #         )
                    self.current_image.current_border = []
            progress.close()
            self.save_current_image(self.save_file)

    def identify_segments(self):
        #TODO
        NotImplemented