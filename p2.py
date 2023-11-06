import matplotlib.pyplot as plt
import numpy as np
from time import time

"""
TODO: Move to functions
TODO: Change into Monte Carlo
TODO: Separate test
TODO: Separate plot
"""

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


def randoms_in_range(min,max,n):
    """ Returns n random numbers within the [min,max] range specified """
    return ( np.random.rand(n)*(max-min) + min )



def monte_carlo_mandelbrot(samples=1000,iterations=100,method='random sampling'):
    # Area estimation only
    xmin, xmax = -2, 1
    ymin, ymax = -1.5, 1.5

    
    if method == 'random sampling':
        # Get numbers stochastic
        real_parts = randoms_in_range(xmin,xmax,samples)
        imaginary_parts = randoms_in_range(ymin,ymax,samples)
    else:
        # By default get linspace values (not Montecarlo!)
        real_parts =  np.linspace(xmin, xmax, image_size)
        imaginary_parts = np.linspace(ymin, ymax, image_size)       


    R, I = np.meshgrid(real_parts, imaginary_parts)
    c = R + 1j * I

    mandelbrot = check_point(c, iterations)

    fraction = np.sum(np.where(mandelbrot < iterations, 0, 1) / samples ** 2)
    

    return (fraction * np.abs(xmax - xmin) * np.abs(ymax - ymin))





if __name__ == '__main__':

    # Run estimation
    area = monte_carlo_mandelbrot(samples=5000,iterations=100,method='random sampling')
    print(area)

    # TODO: Run with many values
    # TODO: Save the values
    # TODO: Statistical analysis + confidence interval
    # balance i and s so that the errors caused by the finiteness of i and s are comparable


    # TODO: 
    """

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
    """