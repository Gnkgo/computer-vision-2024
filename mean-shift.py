import time
import os
import random
import math
import torch
import numpy as np
from skimage import io, color
from skimage.transform import rescale

# Efficiently compute pairwise distances between points
def distance_batch(X, x):
    # Compute squared Euclidean distance in a batch
    dist = torch.cdist(X, x)  # Result shape: (N, 1)
    return dist

# Gaussian function
def gaussian(dist, bandwidth):
    return torch.exp(-dist / (2 * bandwidth ** 2))

# Update point based on weights
def update_point_batch(weight, X):
    # Weighted mean of the points (new centroid)
    return torch.sum(X * weight[:, None], dim=0) / torch.sum(weight)

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    # Iterate over each point and update it based on mean shift algorithm
    dist = distance_batch(X, X)
    weight = gaussian(dist, bandwidth)
    X_ = update_point_batch(weight, X)
    return X_

def meanshift(X, max_iter=20, bandwidth=2.5):
    X = X.clone()
    for _ in range(max_iter):
        X = meanshift_step_batch(X, bandwidth)
    return X

scale = 0.25  # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape  # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Convert to tensor
X = torch.from_numpy(image_lab).float()

# Run your mean-shift algorithm
t = time.time()
X_result = meanshift(X).detach().cpu().numpy()
t = time.time() - t
print(f'Elapsed time for mean-shift: {t}')

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

# Quantize the resulting points and assign labels
centroids, labels = np.unique((X_result / 4).round(), return_inverse=True, axis=0)

# Create result image from labels
result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)  # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)

# Save the result
io.imsave('result.png', result_image)
