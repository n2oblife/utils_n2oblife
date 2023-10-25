#TODO add some function of computer vision on polygons
import numpy as np

def define_border(image, border_list=None, point:tuple=(0,0))->None:
    """Recursive function which looks around a pixel to add it to the current border research list.
    We assume there is only one pixel around a given mask pixel which is a border.

    Args:
        image (any) : all the data needed
        border_list (array) : array with all the edges already found
        point (tuple) : coordinates of a point
    """
    assert(image.is_border(point)), IndexError(
        f"The pixel from which we should define a border need to be a border pixel : ({x}, {y})")
    if not point in border_list :
        border_list.append(point)
        for val_y in (-1, 0, 1):
            for val_x in (-1, 0, 1):
                if not (val_x==0 and val_y==0):
                    if (0<=point[0]+val_x<image.dim_x 
                        and 0<=point[1]+val_y<image.dim_y):
                        if image.is_border((point[0]+val_x, 
                                      point[1]+val_y)) :
                            define_border((point[0]+val_x, 
                                           point[1]+val_y))

def get_center(edges_list:list=None):
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

def get_angle(center:tuple, point:tuple):
    x = point[0] - center[0]
    y = point[1] - center[1]
    angle = np.arctan2(y,x)
    if angle <= 0 :
        angle = 2*np.pi + angle
    return angle

def get_distance(pt1:tuple, pt2:tuple):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    return np.sqrt(x*x + y*y)