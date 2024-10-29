import numpy as np
import cv2
from matplotlib import pyplot as plt


# set a fixed random seed
np.random.seed(42)

# Load the image
image = cv2.imread('data/data_kmeans/img.jpg')  # Provide your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image into a 2D array of pixels
pixel_values = image_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print("pixel_values.shape", pixel_values.shape)
print("First 5 pixel_values", pixel_values[:5])

# Define the number of clusters (K)
# TODO: try 2, 3, 5, 8, 10 etc.
K = 6 # Set the number of clusters

# Randomly initialize the centroids
# Random select K elements from pixel_values as the initial centroids
centroids = pixel_values[np.random.choice(pixel_values.shape[0], K, replace=False)]
print("centroids.shape", centroids.shape)
print("centroids", centroids)
 # TODO


# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2)) #TODO

# Define the K-means algorithm
def kmeans(pixel_values, centroids, max_iterations=50):
    for i in range(max_iterations):
        # Assign each pixel to the closest centroid
        labels = np.zeros(pixel_values.shape[0], dtype=np.int32)
        for idx, pixel in enumerate(pixel_values):
            # Calculate the distances to the centroids
            distances = np.zeros(K)
            for k in range(K):
                distances[k] = euclidean_distance(centroids[k], pixel)
            # Assign the index of the nearest centroid
            labels[idx] = np.argmin(distances)

        # Recalculate the centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            # Get the cluster_points
            cluster_points = pixel_values[labels == k]  # TODO
            # Calculate the new centroids based on the mean of the assigned cluster_points
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)  # TODO

        # Check for convergence
        # Stop when centroids no longer change
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# Run the K-means algorithm
labels, centroids = kmeans(pixel_values, centroids)

# Map each pixel to a distinct color based on its cluster label
# Define a set of distinct colors for each cluster
colors = np.array([
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [255, 0, 255],  # Magenta
    [0, 255, 255],  # Cyan
    [128, 0, 128],  # Purple
    [255, 165, 0],  # Orange
    [128, 128, 128],  # Gray
    [0, 128, 128],  # Teal
])

# Ensure we only have as many colors as there are clusters
colors = colors[:K]

# Assign colors to each pixel based on the cluster it belongs to
segmented_image = colors[labels]
segmented_image = segmented_image.reshape(image_rgb.shape)

# Show the original and segmented image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f'Segmented Image with K={K}')
plt.show()
