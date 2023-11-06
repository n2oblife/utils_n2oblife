#TODO add some function of computer vision on polygons
import math
import cv2
import numpy as np
import matplotlib as plt
from PIL import Image

def get_color(ind, hex=False):

    colors = [(111, 74, 0),
        (81, 0, 81),
        (128, 64, 128),
        (244, 35, 232),
        (250, 170, 160),
        (230, 150, 140),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (180, 165, 180),
        (150, 100, 100),
        (150, 120, 90),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 0, 90),
        (0, 0, 110),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (0, 0, 142), ]

    color = colors[ind % len(colors)]

    if hex:
        return '#%02x%02x%02x' % (color[0], color[1], color[2])
    else:
        return color

def define_border(image, border_list=None, point:tuple=(0,0))->None:
    """Recursive function which looks around a pixel to add it to the current border research list.
    We assume there is only one pixel around a given mask pixel which is a border.

    Args:
        image (any) : all the data needed
        border_list (array) : array with all the edges already found
        point (tuple) : coordinates of a point
    """
    assert(image.is_border(point)), IndexError(
        f"The pixel from which we should define a border need to be a border pixel : ({point[0]}, {point[1]})")
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


def draw_3d_box(im, verts, color=(0, 200, 200), thickness=1):

    for lind in range(0, verts.shape[0] - 1):
        v1 = verts[lind]
        v2 = verts[lind + 1]
        cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)


def draw_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=(0, 200, 200), scale=1, thickness=2):

    w = l3d * scale
    l = w3d * scale
    x = x3d * scale
    z = z3d * scale
    r = ry3d*-1

    corners1 = np.array([
        [-w / 2, -l / 2, 1],
        [+w / 2, -l / 2, 1],
        [+w / 2, +l / 2, 1],
        [-w / 2, +l / 2, 1]
    ])

    ry = np.array([
        [+math.cos(r), -math.sin(r), 0],
        [+math.sin(r), math.cos(r), 0],
        [0, 0, 1],
    ])

    corners2 = ry.dot(corners1.T).T

    corners2[:, 0] += w/2 + x + canvas_bev.shape[1] / 2
    corners2[:, 1] += l/2 + z

    draw_line(canvas_bev, corners2[0], corners2[1], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[1], corners2[2], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[2], corners2[3], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[3], corners2[0], color=color, thickness=thickness)


def draw_line(im, v1, v2, color=(0, 200, 200), thickness=1):

    cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)


def draw_circle(im, pos, radius=5, thickness=1, color=(250, 100, 100), fill=True):

    if fill: thickness = -1

    cv2.circle(im, (int(pos[0]), int(pos[1])), radius, color=color, thickness=thickness)


def draw_2d_box(im, box, color=(0, 200, 200), thickness=1):

    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    x2 = (x + w) - 1
    y2 = (y + h) - 1

    cv2.rectangle(im, (int(x), int(y)), (int(x2), int(y2)), color, thickness)


def imshow(im, fig_num=None):

    if fig_num is not None: plt.figure(fig_num)

    if len(im.shape) == 2:
        im = np.tile(im, [3, 1, 1]).transpose([1, 2, 0])

    plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR))
    plt.show(block=False)


def imwrite(im, path):

    cv2.imwrite(path, im)


def imread(path):

    return cv2.imread(path)


def draw_tick_marks(im, ticks):

    ticks_loc = list(range(0, im.shape[0] + 1, int((im.shape[0]) / (len(ticks) - 1))))

    for tind, tick in enumerate(ticks):
        y = min(max(ticks_loc[tind], 50), im.shape[0] - 10)
        x = im.shape[1] - 115

        draw_text(im, '-{}m'.format(tick), (x, y), lineType=2, scale=1.1, bg_color=None)


def draw_text(im, text, pos, scale=0.4, color=(0, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1):

    pos = [int(pos[0]), int(pos[1])]

    if bg_color is not None:

        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(pos[0] + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)

        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# adopted from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def mat2euler(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])

    else: raise ValueError('singular matrix found in mat2euler')

    return np.array([x, y, z])


def fig_to_im(fig):

    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    w, h, d = buf.shape

    im_pil = Image.frombytes("RGBA", (w, h), buf.tostring())
    im_np = np.array(im_pil)[:,:,:3]

    return im_np


def imzoom(im, zoom=0):

    # single value passed in for both axis?
    # extend same val for w, h
    zoom = np.array(zoom)
    if zoom.size == 1: zoom = np.array([zoom, zoom])

    zoom = np.clip(zoom, a_min=0, a_max=0.99)

    cx = im.shape[1]/2
    cy = im.shape[0]/2

    w = im.shape[1]*(1 - zoom[0])
    h = im.shape[0]*(1 - zoom[-1])

    x1 = int(np.clip(cx - w / 2, a_min=0, a_max=im.shape[1] - 1))
    x2 = int(np.clip(cx + w / 2, a_min=0, a_max=im.shape[1] - 1))
    y1 = int(np.clip(cy - h / 2, a_min=0, a_max=im.shape[0] - 1))
    y2 = int(np.clip(cy + h / 2, a_min=0, a_max=im.shape[0] - 1))

    im = im[y1:y2+1, x1:x2+1, :]

    return im


def imhstack(im1, im2):

    sf = im1.shape[0] / im2.shape[0]

    if sf > 1:
        im2 = cv2.resize(im2, (int(im2.shape[1] / sf), im1.shape[0]))
    else:
        im1 = cv2.resize(im1, (int(im1.shape[1] / sf), im2.shape[0]))


    im_concat = np.hstack((im1, im2))

    return im_concat


# Calculates Rotation Matrix given euler angles.
# adopted from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def euler2mat(x, y, z):

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]
                    ])

    R_y = np.array([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]
                    ])

    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def convertAlpha2Rot(alpha, z3d, x3d):

    ry3d = alpha + math.atan2(-z3d, x3d) + 0.5 * math.pi
    #ry3d = alpha + math.atan2(x3d, z3d)# + 0.5 * math.pi

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return ry3d


def convertRot2Alpha(ry3d, z3d, x3d):

    alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
    #alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi

    while alpha > math.pi: alpha -= math.pi * 2
    while alpha < (-math.pi): alpha += math.pi * 2

    return alpha