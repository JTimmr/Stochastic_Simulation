"""
Bootstrapping with replacement
But issue of having smallest error with a single subset remains

- Moved to functions
- Added computation and plot to see evolution of error with r
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_pi(xs, ys, m):
    return np.sum(np.where(xs**2 + ys**2 <= 1, 1, 0)) / m * 4

def estimate_pi(n,r):
    """
    Computes estimation of pi with n samples and mse error with r subsets 

    Returns:
    pi
    estimated_mse 
    """

    xs = np.random.sample(n)
    ys =  np.random.sample(n)

    pi = calculate_pi(xs, ys, n)
    Fe = pi

    m = 1000
    estimated_mse = 0

    for i in range(r):
        subset_indices = np.random.choice(n, m, replace=True)
        subset_xs = xs[subset_indices]
        subset_ys = ys[subset_indices]

        g = calculate_pi(subset_xs, subset_ys, m)
        Y = (g - Fe)**2
        estimated_mse += Y / r

    
    return pi, estimated_mse



n = 1_000_000
r = 1

pi,estimated_mse = estimate_pi(n,r)
error = pi - np.pi

print('Individual call')
print('pi',pi,'mse',estimated_mse,'error',error)

# Simple plot for many r values
n = 1_000_000
r = np.linspace(1,100_000,50)
errors = []

for i in range(r.shape[0]):
    print('Iteration',i,'/',r.shape[0])
    pi,estimated_mse = estimate_pi(n,int(r[i]))
    errors.append(pi - np.pi)


plt.plot(r,errors)
plt.tight_layout()
plt.xlabel('r')
plt.ylabel('pi - np.pi')
plt.show()