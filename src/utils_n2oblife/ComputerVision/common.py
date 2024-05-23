import numpy as np

def init_nuc(image):
    return np.ones(len(image), len(image[0])), np.zeros(len(image), len(image[0]))

def build_kernel(image, i, j, k_size=3):
    kernel_im = []
    for l in range(i-k_size, i+k_size):
            for m in range(j-k_size, j+k_size):
                if 0 <= l < len(image) and 0 <= m < len(image[0]):
                    kernel_im.append(image[l,m])
    return kernel_im