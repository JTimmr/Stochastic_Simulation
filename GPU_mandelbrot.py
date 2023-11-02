from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import warnings
from time import time



def color_map1(mandelbrot, threshold):

    # Color Mandelbrot set in tints of black
    color = torch.where(mandelbrot == threshold, torch.tensor(0, device=mandelbrot.device), 1 - (mandelbrot + 1) / threshold)
    return torch.stack((color, color, color), dim=-1).cpu().numpy()


def color_map2(mandelbrot, threshold):
    # Color Mandelbrot set in tints of white and green
    color = torch.where(mandelbrot == threshold, torch.tensor(0, device=mandelbrot.device), 1 - (mandelbrot + 1) / threshold)
    red = torch.where(color > 0.5, color, torch.tensor(1, device=mandelbrot.device))
    green = torch.where(color > 0.5, torch.tensor(1, device=mandelbrot.device), torch.tensor(1, device=mandelbrot.device))
    blue = torch.where(color > 0.5, color, torch.tensor(1, device=mandelbrot.device))

    return torch.stack((red, green, blue), dim=-1).cpu().numpy()


def color_map3(mandelbrot, threshold):
    # Color Mandelbrot set in tints of white, black and green
    color = torch.where(mandelbrot == threshold, torch.tensor(0, device=mandelbrot.device), 1 - (mandelbrot + 1) / threshold)
    red = torch.where(color > 0.5, color, torch.tensor(0, device=mandelbrot.device))
    green = torch.where(color > 0.5, torch.tensor(1, device=mandelbrot.device), torch.tensor(0, device=mandelbrot.device))
    blue = torch.where(color > 0.5, color, torch.tensor(0, device=mandelbrot.device))

    return torch.stack((red, green, blue), dim=-1).cpu().numpy()


def check_points(coordinates, threshold):

    # Make PyTorch tensor on GPU to store data
    z = torch.zeros_like(coordinates, dtype=torch.complex64, device=coordinates.device)
    n = torch.zeros(coordinates.shape, dtype=torch.int32, device=coordinates.device)

    # Calculate for each coordinate in the tensor how fast it diverges, if at all
    for _ in range(threshold):
        z = z * z + coordinates
        mask = torch.abs(z) <= 2
        n += mask.to(torch.int32)
    return n


def generate_chunks(chunk_size, image_size, threshold, real_parts, imaginary_parts):
    num_chunks = 0

    # Loop over individual chunks
    for i in range(0 * chunk_size, image_size, chunk_size):
        for j in range(0, image_size, chunk_size):
            num_chunks += 1

            # Extract evaluated coordinates with respect to current chunk
            real_parts_chunk = real_parts[i:i + chunk_size]
            imaginary_parts_chunk = imaginary_parts[j:j + chunk_size]
            R, I = np.meshgrid(real_parts_chunk, imaginary_parts_chunk)

            # Create PyTorch Tensor on GPU containing all coordinates to be evaluated
            coordinates = torch.tensor(R, dtype=torch.float32, device='cuda') + 1j * torch.tensor(I, dtype=torch.float32, device='cuda')

            # Check for each coordinate how many iterations it takes before it diverges
            mandelbrot = check_points(coordinates, threshold)

            # Color the inspected region according to divergence rate
            colors = color_map3(mandelbrot, threshold)

            # Plot and save region
            plt.figure(figsize=(chunk_size / 100, chunk_size / 100), dpi=100)
            plt.axis('off')
            plt.imshow(colors, extent=(-2, 1, -1.5, 1.5))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            plt.savefig(os.path.join(folder_path, f'mandelbrot_chunk_{int(i/chunk_size)}_{int(j/chunk_size)}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

    return num_chunks


def combine_chunks_and_save(folder_path, file_pattern, num_chunks, output_file):

    # Load the chunks
    images = [file_pattern.format(i, j) for i in range(num_chunks) for j in range(num_chunks)]
    chunks = [Image.open(os.path.join(folder_path, image)) for image in images]

    # Set sizes and make blank final image
    chunk_size = chunks[0].size
    image_size = (chunk_size[0] * num_chunks, chunk_size[1] * num_chunks)
    final_image = Image.new('RGB', image_size)

    # Paste chunks in correct place in the final image
    for i in range(0, image_size[0], chunk_size[0]):
        for j in range(0, image_size[1], chunk_size[1]):
            final_image.paste(chunks.pop(0), (i, j))

    final_image.save(os.path.join(folder_path, output_file))


def get_area(xmin, xmax, ymin, ymax, resolution, threshold):
    real_parts = np.linspace(xmin, xmax, resolution)
    imaginary_parts = np.linspace(ymin, ymax, resolution)

    R, I = np.meshgrid(real_parts, imaginary_parts)

    # Create PyTorch Tensor on GPU containing all coordinates to be evaluated
    coordinates = torch.tensor(R, dtype=torch.float32, device='cuda') + 1j * torch.tensor(I, dtype=torch.float32, device='cuda')

    # Check for each coordinate how many iterations it takes before it diverges
    mandelbrot = check_points(coordinates, threshold)

    fraction = torch.sum(torch.where(mandelbrot < n_threshold, torch.tensor(0, device=mandelbrot.device), torch.tensor(1, device=mandelbrot.device)) / image_size ** 2)
    print(fraction * np.abs(xmax - xmin) * np.abs(ymax - ymin))




if __name__ == '__main__':

    start = time()
    # Parameters for generating the Mandelbrot set
    n_threshold = 100
    image_size = 10000
    chunk_size = 10000

    # Investigated region
    ymin, ymax = -1.5, 1.5
    xmin, xmax = -2, 1

    # ymin, ymax = -0.1, -0.092
    # xmin, xmax = -0.751, -0.745

    # Generate chunks of the Mandelbrot set
    make_chunks = False

    # Combine chunks to a single PNG image
    combine_chunks = False

    # Ignore system warnings about resource management. Set to True at own risk!
    potentially_break_machine = True

    calculate_surface = True

    # Dummy variable
    num_chunks = int((image_size/chunk_size) ** 2)


    if potentially_break_machine:
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)

    if calculate_surface:
        resolution = image_size
        get_area(xmin, xmax, ymin, ymax, resolution, n_threshold)

    if make_chunks:
        real_parts = np.linspace(xmin, xmax, image_size)
        imaginary_parts = np.linspace(ymin, ymax, image_size)

        folder_path = "mandelbrot_chunks"
        os.makedirs(folder_path, exist_ok=True)

        num_chunks = generate_chunks(chunk_size, image_size, n_threshold, real_parts, imaginary_parts)

    if combine_chunks:
        combine_chunks_and_save(folder_path, "mandelbrot_chunk_{}_{}.png", num_chunks=int(num_chunks**0.5), output_file="mandelbrot_final.png")


    end = time()

    print(end - start)