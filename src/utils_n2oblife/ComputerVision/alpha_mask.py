# This file is made as lib to help constructing alpha mask
# This script takes an image as input and checks if it 

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_input():
    parser = ArgumentParser(
        description="Finalizing the alpha frame to use.",
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input')
    parser.add_argument('--output', default='output.png')
    return vars(parser.parse_args())

def get_max_pix(data:np.array):
    pix_values = np.zeros(256)
    for line in data:
        for pix in line:
            try:
                max_pix = max(pix)
            except TypeError:
                max_pix = pix
            pix_values[max_pix] += 1
    return pix_values



def data_reparition(data:np.array, title=""):
    """Funciton to show the data repartition of a 2D array

    Args:
        data (np.array): The 2D array from which to show 
    """
    all_pix_values = np.arange(0, 256)
    data_pix_values = get_max_pix(data)
    plt.plot(all_pix_values, data_pix_values)
    plt.title(title)
    plt.show()

def finalize_mask(image:Image, filter=250):
    pixels = image.load() # create a pixels map
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            pix = pixels[i,j]
            try:
                white_pix = (max(pixels[i,j]) >= filter)
                RGB = True
            except:
                white_pix = (pixels[i,j] >= filter)
                RGB = False
            if RGB:
                if white_pix:
                    pixels[i,j] = (255,255,255)
                else:
                    pixels[i,j] = (0,0,0)
            else:
                if white_pix:
                    pixels[i,j] = 255
                else:
                    pixels[i,j] = 0
    # data = np.asarray(image)
    # data_reparition(data, "test")


if __name__ == '__main__':
    # Parse args
    vars = parse_input()
    input = vars['input']
    output = vars['output']
    image = Image.open(input)
    finalize_mask(image, 240)
    image.save('output.png')
    data_reparition(np.asarray(image))

            