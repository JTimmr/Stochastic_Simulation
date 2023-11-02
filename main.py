import matplotlib.pyplot as plt
import numpy as np
from time import time

def color_map1(n, threshold):
    color = np.where(n == threshold, 0, 1 - (n + 1) / (threshold))
    return np.stack((color, color, color), axis=-1)


def color_map2(n, threshold):
    color = np.where(n == threshold, 0, 1 - (n + 1) / threshold)
    white = np.where(n == threshold, 0, 1)
    red = np.where(color > 0.5, color, white)
    green = np.where(color > 0.5, 1, white)
    blue = np.where(color > 0.5, color, white)

    return np.stack((red, green, blue), axis=-1)


def check_point(c, threshold):
    z = np.zeros_like(c)
    n = np.zeros(c.shape)
    for _ in range(threshold):
        z = z ** 2 + c
        mask = np.abs(z) <= 2
        n += mask
    return n



if __name__ == '__main__':

    plot_set = False
    calculate_surface = True

    start = time()

    # Parameters for generating the Mandelbrot set
    n_threshold = 100
    image_size = 10000

    # Investigated region
    ymin, ymax = -1.5, 1.5
    xmin, xmax = -2, 1

    # ymin, ymax = -0.75, -0.6
    # xmin, xmax = -0.4, -0.25

    real_parts = np.linspace(xmin, xmax, image_size)
    imaginary_parts = np.linspace(ymin, ymax, image_size)

    R, I = np.meshgrid(real_parts, imaginary_parts)

    c = R + 1j * I

    mandelbrot = check_point(c, n_threshold)

    if calculate_surface:

        fraction = np.sum(np.where(mandelbrot < n_threshold, 0, 1) / image_size ** 2)
        print(fraction * np.abs(xmax - xmin) * np.abs(ymax - ymin))

    if plot_set:
        colors = color_map2(mandelbrot, n_threshold)

        plt.imshow(colors, extent=(xmin, xmax, ymin, ymax))

        plt.show()

    end = time()

    print(end - start)