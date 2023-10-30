import matplotlib.pyplot as plt
import numpy as np
from time import time

def make_color(n, threshold):
    # case 1: nr is in mandelbrot set
    if n == threshold:
        return (0.9,0.1,0.1)
    else:
        color = n / (threshold)
        return (color, color, color)
        

def check_point(c, threshold):
    z = 0
    n = 0
    while abs(z) < 10 and n < threshold:
        addition = z ** 2 + c
        z += addition
        n += 1
    # Increase i of numbers in mandelbrot set
    # numbers with i 1 over the limit will be plotted accordingly
    # if abs(z) < 50:
    #     i +=1

    return make_color(n, threshold)

c = complex(0,0)

threshold = 10
print(check_point(c, threshold))

real_parts = np.linspace(-3,1,50)
imaginary_parts = np.linspace(-2,2,50)

plt.figure()


for real in real_parts:
    for imaginary in imaginary_parts:
        c = complex(real,imaginary)
        color= check_point(c, threshold)
        plt.scatter(real,imaginary, color=color)

plt.show()