import matplotlib.pyplot as plt
import numpy as np

def color_map(n, max_iter):
    color = np.where(n == max_iter, 0, 1 - (n + 1) / (max_iter))
    return np.stack((color, color, color), axis=-1)
        

def check_point(c, threshold):
    z = np.zeros_like(c)
    n = np.zeros(c.shape)
    for _ in range(threshold):
        z = z ** 2 + c
        mask = np.abs(z) <= 2
        n += mask
    return n

threshold = 100

real_parts = np.linspace(-2,1,10000)
imaginary_parts = np.linspace(-1.5,1.5,10000)

R, I = np.meshgrid(real_parts, imaginary_parts)

c = R + 1j * I

mandelbrot = check_point(c, threshold)

colors = color_map(mandelbrot, threshold)

plt.imshow(colors, extent=(-2, 1, -1.5, 1.5))

plt.show()