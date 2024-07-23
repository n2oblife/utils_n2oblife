Here's the updated code with comments explaining each function and its parameters:

```python
# TODO add some functions of computer vision on polygons
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_color(name, hex=False):
    """
    Get a color from a predefined dictionary of colors.

    Args:
        name (str): Name of the color to select.
        hex (bool): If True, returns the color in hexadecimal format.

    Returns:
        tuple or str: RGB color tuple or hexadecimal string.
    """
    colors = {
        "brown": (111, 74, 0),
        "purple": (81, 0, 81),
        "violet": (128, 64, 128),
        "pink": (244, 35, 232),
        "light_pink": (250, 170, 160),
        "peach": (230, 150, 140),
        "dark_gray": (70, 70, 70),
        "grayish_blue": (102, 102, 156),
        "light_brown": (190, 153, 153),
        "pale_brown": (180, 165, 180),
        "warm_brown": (150, 100, 100),
        "beige": (150, 120, 90),
        "gray": (153, 153, 153),
        "yellow": (250, 170, 30),
        "bright_yellow": (220, 220, 0),
        "dark_green": (107, 142, 35),
        "light_green": (152, 251, 152),
        "blue": (70, 130, 180),
        "red": (220, 20, 60),
        "bright_red": (255, 0, 0),
        "dark_blue": (0, 0, 142),
        "darker_blue": (0, 0, 70),
        "medium_blue": (0, 60, 100),
        "navy_blue": (0, 0, 90),
        "deep_blue": (0, 0, 110),
        "teal": (0, 80, 100),
        "royal_blue": (0, 0, 230),
        "dark_red": (119, 11, 32)
    }

    if name not in colors:
        raise ValueError(f"Color '{name}' not found in the predefined colors.")

    color = colors[name]

    if hex:
        return '#%02x%02x%02x' % (color[0], color[1], color[2])
    else:
        return color


def define_border(image, border_list=None, point: tuple = (0, 0)) -> None:
    """
    Recursive function to find the border of a region in an image starting from a given point.

    Args:
        image (any): Image object containing the data.
        border_list (list): List to store the border points.
        point (tuple): Starting point coordinates.

    Raises:
        IndexError: If the given point is not a border pixel.
    """
    assert image.is_border(point), IndexError(
        f"The pixel from which we should define a border need to be a border pixel : ({point[0]}, {point[1]})")
    if point not in border_list:
        border_list.append(point)
        for val_y in (-1, 0, 1):
            for val_x in (-1, 0, 1):
                if not (val_x == 0 and val_y == 0):
                    if 0 <= point[0] + val_x < image.dim_x and 0 <= point[1] + val_y < image.dim_y:
                        if image.is_border((point[0] + val_x, point[1] + val_y)):
                            define_border(image, border_list, (point[0] + val_x, point[1] + val_y))

def get_center(edges_list: list = None) -> tuple:
    """
    Calculate the center of a polygon defined by its edges.

    Args:
        edges_list (list): List of edge coordinates (tuples).

    Returns:
        tuple: Coordinates of the polygon's center.
    """
    sx = sy = sL = 0
    for i in range(len(edges_list)):
        x0, y0 = edges_list[i - 1]
        x1, y1 = edges_list[i]
        L = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
        sx += (x0 + x1) / 2 * L
        sy += (y0 + y1) / 2 * L
        sL += L
    return int(sx // sL), int(sy // sL)

def get_angle(center: tuple, point: tuple) -> float:
    """
    Calculate the angle between the center of a polygon and a given point.

    Args:
        center (tuple): Center coordinates.
        point (tuple): Point coordinates.

    Returns:
        float: Angle in radians.
    """
    x = point[0] - center[0]
    y = point[1] - center[1]
    angle = np.arctan2(y, x)
    if angle <= 0:
        angle = 2 * np.pi + angle
    return angle

def get_distance(pt1: tuple, pt2: tuple) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        pt1 (tuple): First point coordinates.
        pt2 (tuple): Second point coordinates.

    Returns:
        float: Distance between the points.
    """
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    return np.sqrt(x **2 + y **2)

def draw_3d_box(im, verts, color=(0, 200, 200), thickness=1):
    """
    Draw a 3D box on an image.

    Args:
        im (ndarray): Image on which to draw.
        verts (ndarray): Vertices of the 3D box.
        color (tuple): Color of the box lines.
        thickness (int): Thickness of the lines.
    """
    for lind in range(0, verts.shape[0] - 1):
        v1 = verts[lind]
        v2 = verts[lind + 1]
        cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)

def draw_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=(0, 200, 200), scale=1, thickness=2):
    """
    Draw a bird's eye view (BEV) box on a canvas.

    Args:
        canvas_bev (ndarray): Canvas image for BEV.
        z3d (float): Z-coordinate of the 3D box.
        l3d (float): Length of the 3D box.
        w3d (float): Width of the 3D box.
        x3d (float): X-coordinate of the 3D box.
        ry3d (float): Rotation around the Y-axis.
        color (tuple): Color of the box lines.
        scale (float): Scaling factor for the box dimensions.
        thickness (int): Thickness of the lines.
    """
    # Scale the dimensions and coordinates of the box
    w = l3d * scale
    l = w3d * scale
    x = x3d * scale
    z = z3d * scale
    r = ry3d * -1  # Negate rotation angle for clockwise rotation

    # Define the corners of the box before rotation and translation
    corners1 = np.array([
        [-w / 2, -l / 2, 1],
        [+w / 2, -l / 2, 1],
        [+w / 2, +l / 2, 1],
        [-w / 2, +l / 2, 1]
    ])

    # Define the rotation matrix for rotating around the Z-axis
    ry = np.array([
        [+math.cos(r), -math.sin(r), 0],
        [+math.sin(r), math.cos(r), 0],
        [0, 0, 1],
    ])

    # Apply the rotation to the corners of the box
    corners2 = ry.dot(corners1.T).T

    # Translate the rotated corners to the correct position on the canvas
    corners2[:, 0] += w / 2 + x + canvas_bev.shape[1] / 2
    corners2[:, 1] += l / 2 + z

    # Draw the edges of the box on the canvas
    draw_line(canvas_bev, corners2[0], corners2[1], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[1], corners2[2], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[2], corners2[3], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[3], corners2[0], color=color, thickness=thickness)


def draw_line(im, v1, v2, color=(0, 200, 200), thickness=1):
    """
    Draw a line between two points on an image.

    Args:
        im (ndarray): Image on which to draw.
        v1 (tuple): Starting point coordinates.
        v2 (tuple): Ending point coordinates.
        color (tuple): Color of the line.
        thickness (int): Thickness of the line.
    """
    cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)

def draw_circle(im, pos, radius=5, thickness=1, color=(250, 100, 100), fill=True):
    """
    Draw a circle on an image.

    Args:
        im (ndarray): Image on which to draw.
        pos (tuple): Center position of the circle.
        radius (int): Radius of the circle.
        thickness (int): Thickness of the circle edge. Use -1 for filled circle.
        color (tuple): Color of the circle.
        fill (bool): If True, draw a filled circle.
    """
    if fill:
        thickness = -1
    cv2.circle(im, (int(pos[0]), int(pos[1])), radius, color=color, thickness=thickness)

def draw_2d_box(im, box, color=(0, 200, 200), thickness=1):
    """
    Draw a 2D bounding box on an image.

    Args:
        im (ndarray): Image on which to draw.
        box (tuple): Bounding box coordinates (x, y, width, height).
        color (tuple): Color of the box.
        thickness (int): Thickness of the box lines.
    """
    x, y, w, h = box
    x2 = (x + w) - 1
    y2 = (y + h) - 1
    cv2.rectangle(im, (int(x), int(y)), (int(x2), int(y2)), color, thickness)

def imshow(im, fig_num=None):
    """
    Display an image using matplotlib.

    Args:
        im (ndarray): Image to display.
        fig_num (int, optional): Figure number for matplotlib.
    """
    if fig_num is not None:
        plt.figure(fig_num)

    if len(im.shape) == 2:
        im = np.tile(im, [3, 1, 1]).transpose([1, 2, 0])

    plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR))
    plt.show(block=False)

def imwrite(im, path):
    """
    Save an image to a file.

    Args:
        im (ndarray): Image to save.
        path (str): File path to save the image.
    """
    cv2.imwrite(path, im)

def imread(path):
    """
    Read an image from a file.

    Args:
        path (str): File path to read the image from.

    Returns:
        ndarray: Image read from the file.
    """
    return cv2.imread(path)

def draw_tick_marks(im, ticks):
    """
    Draw tick marks on an image.

    Args:
        im (ndarray): Image on which to draw.
        ticks (list): List of tick values.
    """
    ticks_loc = list(range(0, im.shape[0] + 1, int(im.shape[0] / (len(ticks) - 1))))

    for tind, tick in enumerate(ticks):
        y = min(max(ticks_loc[tind], 50), im.shape[0] - 10)
        x = im.shape[1] - 115
        draw_text(im, '-{}m'.format(tick), (x, y), lineType=2, scale=1.1, bg_color=None)

def draw_text(im, text, pos, scale=0.4, color=(0, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1):
    """
    Draw text on an image with optional background color.

    Args:
        im (ndarray): Image on which to draw.
        text (str): Text to draw.
        pos (tuple): Position to draw the text.
        scale (float): Scale of the text.
        color (tuple): Color of the text.
        font (int): Font type.
        bg_color (tuple): Background color of the text.
        blend (float): Blending factor for background color.
        lineType (int): Line type for text.
    """
    pos = [int(pos[0]), int(pos[1])]

    if bg_color is not None:
        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(pos[0] + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0] * blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1] * blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2] * blend + bg_color[2] * (1 - blend)

        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)

def mat2euler(R):
    """
    Convert a rotation matrix to Euler angles.

    Args:
        R (ndarray): Rotation matrix.

    Returns:
        ndarray: Euler angles.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        raise ValueError('singular matrix found in mat2euler')

    return np.array([x, y, z])

def fig_to_im(fig):
    """
    Convert a matplotlib figure to an image.

    Args:
        fig (Figure): Matplotlib figure.

    Returns:
        ndarray: Image representation of the figure.
    """
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb gives pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    im_pil = Image.frombytes("RGBA", (w, h), buf.tobytes())
    im_np = np.array(im_pil)[:, :, :3]

    return im_np

def imzoom(im, zoom=0):
    """
    Zoom into an image.

    Args:
        im (ndarray): Image to zoom into.
        zoom (float or list): Zoom factor(s) for width and height.

    Returns:
        ndarray: Zoomed image.
    """
    zoom = np.array(zoom)
    if zoom.size == 1:
        zoom = np.array([zoom, zoom])

    zoom = np.clip(zoom, a_min=0, a_max=0.99)

    cx = im.shape[1] / 2
    cy = im.shape[0] / 2

    w = im.shape[1] * (1 - zoom[0])
    h = im.shape[0] * (1 - zoom[1])

    x1 = int(np.clip(cx - w / 2, a_min=0, a_max=im.shape[1] - 1))
    x2 = int(np.clip(cx + w / 2, a_min=0, a_max=im.shape[1] - 1))
    y1 = int(np.clip(cy - h / 2, a_min=0, a_max=im.shape[0] - 1))
    y2 = int(np.clip(cy + h / 2, a_min=0, a_max=im.shape[0] - 1))

    im = im[y1:y2 + 1, x1:x2 + 1, :]

    return im

def imhstack(im1, im2):
    """
    Horizontally stack two images.

    Args:
        im1 (ndarray): First image.
        im2 (ndarray): Second image.

    Returns:
        ndarray: Concatenated image.
    """
    sf = im1.shape[0] / im2.shape[0]

    if sf > 1:
        im2 = cv2.resize(im2, (int(im2.shape[1] / sf), im1.shape[0]))
    else:
        im1 = cv2.resize(im1, (int(im1.shape[1] / sf), im2.shape[0]))

    im_concat = np.hstack((im1, im2))

    return im_concat

def euler2mat(x, y, z):
    """
    Calculate rotation matrix given Euler angles.

    Args:
        x (float): Rotation around the X-axis.
        y (float): Rotation around the Y-axis.
        z (float): Rotation around the Z-axis.

    Returns:


        ndarray: Rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]])

    R_y = np.array([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]])

    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def convertAlpha2Rot(alpha, z3d, x3d):
    """
    Convert alpha angle to rotation angle.

    Args:
        alpha (float): Alpha angle.
        z3d (float): Z coordinate in 3D space.
        x3d (float): X coordinate in 3D space.

    Returns:
        float: Rotation angle.
    """
    ry3d = alpha + math.atan2(-z3d, x3d) + 0.5 * math.pi

    while ry3d > math.pi:
        ry3d -= math.pi * 2
    while ry3d < -math.pi:
        ry3d += math.pi * 2

    return ry3d

def convertRot2Alpha(ry3d, z3d, x3d):
    """
    Convert rotation angle to alpha angle.

    Args:
        ry3d (float): Rotation angle.
        z3d (float): Z coordinate in 3D space.
        x3d (float): X coordinate in 3D space.

    Returns:
        float: Alpha angle.
    """
    alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi

    while alpha > math.pi:
        alpha -= math.pi * 2
    while alpha < -math.pi:
        alpha += math.pi * 2

    return alpha