
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
print(wine_df.head())

# No missing values?
missing_values = wine_df.isnull().sum()

# Feature scaling (Standardization)
scaler = StandardScaler()
scaled_wine_data = scaler.fit_transform(wine_df)

# Splitting the data
X_train, X_test = train_test_split(scaled_wine_data, test_size=0.2, random_state=42)
missing_values, X_train[:5], X_test[:5]

def pca_from_scratch(X, n_components):
    """
    Implement PCA from scratch.
    X: ndarray, the data
    n_components: int, number of principal components to keep
    """
    cov_matrix = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]

    selected_eigenvectors = sorted_eigenvectors[:, :n_components]

    transformed_data = np.dot(X, selected_eigenvectors)
    
    return transformed_data, sorted_eigenvalues, sorted_eigenvectors

dummytest = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
pca_result, eigenvalues, eigenvectors = pca_from_scratch(dummytest, 2)
pca_result, eigenvalues, eigenvectors

pca_transformed_train, eigenvalues, eigenvectors = pca_from_scratch(X_train, 2)
print("Shape of reduced Wine data:", pca_transformed_train.shape)

# first two features of the original (scaled) training data
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Original Data')
plt.xlabel('First Feature')
plt.ylabel('Second Feature')
plt.title('First Two Features of Original (Scaled) Data')
plt.legend()
plt.show()

# compressed data
plt.figure(figsize=(10, 6))
plt.scatter(pca_transformed_train[:, 0], pca_transformed_train[:, 1], c='red', label='PCA Compressed Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Data Compressed with PCA')
plt.legend()
plt.show()

def svd_from_scratch(X, k):
    """
    Implement Singular Value Decomposition from scratch.
    X: ndarray, the data
    k: int, number of singular values to keep
    """
    covariance_matrix = np.dot(X.T, X) / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_index = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, sorted_index[:k]]
    singular_values = np.sqrt(sorted(eigenvalues, reverse=True)[:k])

    S = np.diag(singular_values)
    U = np.dot(X, selected_eigenvectors)
    U = U / np.linalg.norm(U, axis=0)
    
    return U, S, selected_eigenvectors.T

U, S, Vt = svd_from_scratch(X_train, 2)
U.shape, S.shape, Vt.shape


# compressed data with SVD
plt.figure(figsize=(10, 6))
plt.scatter(U[:, 0], U[:, 1], c='green', label='SVD Compressed Data')
plt.xlabel('First Singular Vector')
plt.ylabel('Second Singular Vector')
plt.title('Data Compressed with SVD')
plt.legend()
plt.show()

# Measuring time and reconstruction error for PCA
start_time_pca = time.time()
pca_transformed_data_test = np.dot(X_test, eigenvectors[:, :2])
pca_reconstructed_data_test = np.dot(pca_transformed_data_test, eigenvectors[:, :2].T)
pca_reconstruction_error_test = np.mean((X_test - pca_reconstructed_data_test) ** 2)
end_time_pca = time.time()
pca_time = end_time_pca - start_time_pca

# Measuring time and reconstruction error for SVD
start_time_svd = time.time()
svd_transformed_data_test = np.dot(X_test, Vt.T)
svd_reconstructed_data_test = np.dot(svd_transformed_data_test, Vt)
svd_reconstruction_error_test = np.mean((X_test - svd_reconstructed_data_test) ** 2)
end_time_svd = time.time()
svd_time = end_time_svd - start_time_svd

print("PCA Reconstruction Error on Test Set:", pca_reconstruction_error_test)
print("PCA Time Taken:", pca_time)
print("SVD Reconstruction Error on Test Set:", svd_reconstruction_error_test)
print("SVD Time Taken:", svd_time)


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def im2gnp(image):
    return np.array(image.convert('L'))

def imshow_gray(im, ax=None):
    if ax is None:
        f = plt.figure()
        ax = plt.axes()
    ax.imshow(im, interpolation='nearest', cmap=plt.get_cmap('gray'))

# Load and resize
original_image_path = "svd_img_original.jpg"
original_image = Image.open(original_image_path)
resize_dims = (300, 300)
resized_image = original_image.resize(resize_dims)
resized_image_np = im2gnp(resized_image)

# Compression: SVD from scratch
def compress_image(I, k):
    U, S, VT = np.linalg.svd(I, full_matrices=False)
    Uk = U[:, :k]
    VkT = VT[:k, :]
    return S, Uk, VkT

# Metrics + Reconstruction
def sizeof_image(I):
    return np.size(I)

def sizeof_compressed_image(S, U, Vt):
    k = U.shape[1]
    return 8 * (U.size + Vt.size + k)

def compression_error(S, k):
    return np.sqrt(np.sum(S[k:] ** 2) / np.sum(S ** 2))

def uncompress_image(S, U, Vt):
    k = U.shape[1]
    S_diag = np.diag(S[:k])
    return np.dot(U, np.dot(S_diag, Vt))

# Find best k
def find_rank(rel_err_target, Sigma):
    sigma_sq = np.sum(Sigma**2)
    cumsum = np.flip(np.cumsum(np.flip(Sigma**2)))
    target = np.argmin(np.sqrt(cumsum / sigma_sq) >= rel_err_target)
    return target

# Compression
k = 50  # Starting value
Sigma, Uk, VkT = compress_image(resized_image_np, k)

# Metrics
original_pixels = sizeof_image(resized_image_np)
compressed_pixels = sizeof_compressed_image(Sigma, Uk, VkT)
compression_ratio = original_pixels / compressed_pixels
error = compression_error(Sigma, k)

print(f"Size of the original image  for Dolphins Image: {original_pixels} pixels")
print(f"Size of the compressed image  for Dolphins Image: {compressed_pixels} pixels")
print(f"Compression Ratio  for Dolphins Image: {compression_ratio}")
print(f"Reconstruction Error for Dolphins Image: {error}")

# Reconstruct
reconstructed_image = uncompress_image(Sigma, Uk, VkT)

# Compare original - reconstructed
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
imshow_gray(resized_image_np, ax=axs[0])
axs[0].set_title('Dolphins Original Image')
imshow_gray(reconstructed_image, ax=axs[1])
axs[1].set_title('Dolphins Reconstructed Image')
plt.show()

# Find best k
rel_err_target = 0.10  # 10% error
optimal_k = find_rank(rel_err_target, Sigma)
print(f"Optimal k for {rel_err_target*100}% error in Dolphins Image is {optimal_k}")


new_image_path = "svd_original.jpg"
new_image = Image.open(new_image_path)
new_resized_image = new_image.resize(resize_dims)
new_resized_image_np = im2gnp(new_resized_image)

new_Sigma, new_Uk, new_VkT = compress_image(new_resized_image_np, k)

new_original_pixels = sizeof_image(new_resized_image_np)
new_compressed_pixels = sizeof_compressed_image(new_Sigma, new_Uk, new_VkT)
new_compression_ratio = new_original_pixels / new_compressed_pixels
new_error = compression_error(new_Sigma, k)

print(f"Size of the Boat image: {new_original_pixels} pixels")
print(f"Size of the Boat compressed image: {new_compressed_pixels} pixels")
print(f"Compression Ratio for Boat image: {new_compression_ratio}")
print(f"Reconstruction Error for Boat image: {new_error}")

new_reconstructed_image = uncompress_image(new_Sigma, new_Uk, new_VkT)

fig, axs = plt.subplots(1, 2, figsize=(10, 10))
imshow_gray(new_resized_image_np, ax=axs[0])
axs[0].set_title('Boat Original Image')
imshow_gray(new_reconstructed_image, ax=axs[1])
axs[1].set_title('Boat Reconstructed Image')
plt.show()

new_optimal_k = find_rank(rel_err_target, new_Sigma)
print(f"Optimal k for {rel_err_target*100}% error in Boat image is {new_optimal_k}")