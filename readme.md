# Same title as report has?

## Features

**1. Surface Area Estimation**
- Utilize Monte Carlo estimation to calculate the surface area of the Mandelbrot set.
- Implement grid, random and Latin Hypercube sampling techniques for thorough exploration.

**2. Convergence Investigation**
- Explore the convergence of the area estimation as a function of the number of iterations in the Mandelbrot calculation.
- Analyze how the choice of sampling method impacts convergence.

**3. GPU and CPU Implementation**
- Leverage GPU acceleration for performance gains when available.
- Automatically switch to CPU if GPU is not accessible.

**4. 95% Confidence Interval**
- Calculate the 95% confidence interval for the area estimation to assess the reliability of the results as a function of the amount of runs.


## How to Use

1. **Clone the repository:**

   ```bash
   git clone https://github.com/JTimmr/Stochastic_Simulation
   cd Stochastic_Simulation

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Open and run the Jupyter Notebook:**
    ```bash
    jupyter notebook main.ipynb


## Important Note

The parameter values provided in the Jupyter Notebook are optimized for GPU programming. If you are running the code on a CPU, be aware that the computation may take a significantly longer time. Consider adjusting the parameters based on your hardware specifications for optimal performance. If CUDA is installed and a GPU is detected, the major part of the code will automatically be executed on the GPU.



This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](LICENSE.txt) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
