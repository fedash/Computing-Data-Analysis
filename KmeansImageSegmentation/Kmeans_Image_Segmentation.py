
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Synthetic Dataset Generation
synthetic_data, synthetic_labels = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
synthetic_df = pd.DataFrame(synthetic_data, columns=['Feature_1', 'Feature_2'])

# Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# Data Exploration
print("=== Synthetic Dataset ===")
print(synthetic_df.head())
print("\n=== Iris Dataset ===")
print(iris_df.head())


# Initialize centroids randomly from the data points
def initialize_centroids(X, K):
    random_indices = np.random.choice(X.shape[0], size=K, replace=False)
    return X[random_indices]

# Assign each data point to the nearest centroid
def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Update the centroids based on cluster assignments
def update_centroids(X, clusters, K):
    new_centroids = np.array([X[clusters == k].mean(axis=0) for k in range(K)])
    return new_centroids

# K-means 
def kmeans_from_scratch(X, K, max_iters=100):
    # Step 1: Initialize centroids randomly from the data points
    centroids = initialize_centroids(X, K)
    
    for i in range(max_iters):
        # Step 2.1: Assignment Step
        clusters = assign_clusters(X, centroids)
        
        # Step 2.2: Update Step
        new_centroids = update_centroids(X, clusters, K)
        for k in range(K):
            points_in_cluster = X[clusters == k]
            # Check if the cluster has points. If not, keep the old centroid.
            if points_in_cluster.shape[0] > 0:
                new_centroids[k] = points_in_cluster.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    inertia = np.sum([np.linalg.norm(X[i] - centroids[clusters[i]])**2 for i in range(len(X))])
    return centroids, clusters, inertia


# Run K-means from scratch
centroids_synthetic, clusters_synthetic, _ = kmeans_from_scratch(synthetic_data, 3)

plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=clusters_synthetic, cmap='viridis')
plt.scatter(centroids_synthetic[:, 0], centroids_synthetic[:, 1], s=300, c='red')
plt.title('K-means from Scratch on Synthetic Data')
plt.show()

centroids_iris, clusters_iris, _ = kmeans_from_scratch(iris_data.data, 3)

plt.scatter(iris_data.data[:, 0], iris_data.data[:, 1], c=clusters_iris, cmap='viridis')
plt.scatter(centroids_iris[:, 0], centroids_iris[:, 1], s=300, c='red')
plt.title('K-means from Scratch on Iris Data')
plt.show()


kmeans_synthetic_sklearn = KMeans(n_clusters=3, n_init=10).fit(synthetic_data)

plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=kmeans_synthetic_sklearn.labels_, cmap='viridis')
plt.scatter(kmeans_synthetic_sklearn.cluster_centers_[:, 0], kmeans_synthetic_sklearn.cluster_centers_[:, 1], s=300, c='red')
plt.title('K-means with sklearn on Synthetic Data')
plt.show()

kmeans_iris_sklearn = KMeans(n_clusters=3, n_init=10).fit(iris_data.data)

plt.scatter(iris_data.data[:, 0], iris_data.data[:, 1], c=kmeans_iris_sklearn.labels_, cmap='viridis')
plt.scatter(kmeans_iris_sklearn.cluster_centers_[:, 0], kmeans_iris_sklearn.cluster_centers_[:, 1], s=300, c='red')
plt.title('K-means with sklearn on Iris Data')
plt.show()



def apply_kmeans_on_image(image, n_clusters, resize_dim=(256, 256)):
    # Resize the image
    image = cv2.resize(image, resize_dim, interpolation=cv2.INTER_AREA)

    # Reshape the image to a 2D pixels array
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Apply K-means
    centroids, clusters, inertia = kmeans_from_scratch(pixels, n_clusters)
    
    # Map each cluster back to centroid
    segmented_pixels = np.array([centroids[cluster] for cluster in clusters])
    segmented_pixels = np.uint8(segmented_pixels)
    
    # Reshape back to the original
    segmented_image = segmented_pixels.reshape(image.shape)
    
    return segmented_image, inertia

original_image = plt.imread('kmeans_original.jpg')

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[0].axis('off')

k_values = [2, 5, 7]

for i, k in enumerate(k_values):
    segmented_image, _ = apply_kmeans_on_image(original_image, n_clusters=k)
    axs[i + 1].imshow(segmented_image)
    axs[i + 1].set_title(f'Segmented Image (k={k})')
    axs[i + 1].axis('off')

plt.show()

def plot_elbow_method(image, max_k=10):
    inertias = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        _, inertia = apply_kmeans_on_image(image, n_clusters=k)
        inertias.append(inertia)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    
    plt.show()

plot_elbow_method(original_image, max_k = 20)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[0].axis('off')

k_values = [4,10,15]

for i, k in enumerate(k_values):
    segmented_image, _ = apply_kmeans_on_image(original_image, n_clusters=k)
    axs[i + 1].imshow(segmented_image)
    axs[i + 1].set_title(f'Segmented Image (k={k})')
    axs[i + 1].axis('off')

plt.show()
