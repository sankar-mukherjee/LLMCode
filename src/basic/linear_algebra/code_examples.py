# Linear Algebra & Machine Learning Implementations
#
# Matrix Operations:
#   - Matrix-Vector Dot Product
#   - Matrix Multiplication
#   - Scalar Multiplication
#   - Mean by Row/Column
#   - 2x2 Matrix Inverse
#
# Statistical Methods:
#   - Covariance Matrix
#   - Principal Component Analysis (PCA)
#   - Power Method (Eigenvalues)
#
# Machine Learning:
#   - Jaccard Index (IoU)
#   - K-Means Clustering

import numpy as np
import torch


def mat_vec_dot(matrix, vector):
    """Matrix-vector multiplication: matrix @ vector"""
    
    # check empty input
    if not matrix or not vector:
        return -1

    rows = len(matrix)
    cols = len(matrix[0])

    # check dimension compatibility
    if cols != len(vector):
        return -1

    result = [0] * rows

    # compute dot product for each row
    for i in range(rows):
        s = 0
        for j in range(cols):
            s += matrix[i][j] * vector[j]
        result[i] = s

    return result

def matmul(A, B):
    """Matrix multiplication: A @ B"""
    r1, c1 = len(A), len(A[0])
    r2, c2 = len(B), len(B[0])

    # check dimension compatibility
    if c1 != r2:
        return -1

    C = [[0]*c2 for _ in range(r1)]

    # compute each element C[i][j]
    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                C[i][j] += A[i][k] * B[k][j]

    return C

def scalar_multiply(A, k):
    """Multiply each element of matrix A by scalar k."""
    rows = len(A)
    cols = len(A[0])

    result = [[0]*cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            result[i][j] = k * A[i][j]

    return result

def mean_matrix(A, axis):
    """
    Compute mean along specified axis.
    axis='row': mean of each row
    axis='col': mean of each column
    """
    rows = len(A)
    cols = len(A[0])

    if axis == "row":
        return [sum(row)/cols for row in A]

    if axis == "col":
        return [sum(A[i][j] for i in range(rows))/rows for j in range(cols)]

def inverse_2x2(A):
    """Compute inverse of 2x2 matrix using analytical formula."""
    a, b = A[0]
    c, d = A[1]

    det = a*d - b*c  # determinant
    if det == 0:
        return -1  # singular matrix (not invertible)

    # inverse formula: (1/det) * [[d,-b],[-c,a]]
    return [[ d/det, -b/det],
            [-c/det,  a/det]]


def covariance_matrix(X):
    """
    Compute covariance matrix from data.
    X : numpy array of shape [samples, features]
    returns: covariance matrix [features × features]
    """
    Xc = X - X.mean(axis=0)            # center each feature
    cov = (Xc.T @ Xc) / (len(X) - 1)   # sample covariance formula
    return cov

def covariance_two_matrices(X, Y):
    """
    X, Y : numpy arrays [samples, features]
    returns cross-covariance matrix
    """
    Xc = X - X.mean(axis=0)   # center features
    Yc = Y - Y.mean(axis=0)

    cov = (Xc.T @ Yc) / (len(X) - 1)
    return cov

def pca(X, k):
    """
    Principal Component Analysis: reduce X to k dimensions.
    
    Args:
        X: numpy array of shape [n_samples, n_features]
        k: number of principal components to keep
    
    Returns:
        X_pca: numpy array of shape [n_samples, k]
    """
    # Compute covariance matrix (centers data internally)
    cov = covariance_matrix(X)       # [n_features, n_features]
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eig(cov)  # eigvals: [n_features], eigvecs: [n_features, n_features]
    
    # Sort indices by variance (descending)
    idx = np.argsort(eigvals)[::-1]  # [n_features]
    
    # Select top-k eigenvectors
    W = eigvecs[:, idx[:k]]          # [n_features, k]
    
    # Center data for projection
    X_centered = X - X.mean(axis=0)  # [n_samples, n_features]
    
    # Project data onto principal components
    return X_centered @ W            # [n_samples, n_features] @ [n_features, k] → [n_samples, k]

def power_method(A, iters=100):
    """
    Find dominant eigenvalue and eigenvector using power iteration.
    Returns: (eigenvalue, eigenvector)
    """
    n = A.shape[0]
    v = np.random.rand(n)  # random initial vector

    # power iteration
    for _ in range(iters):
        v = A @ v
        v = v / np.linalg.norm(v)  # normalize to unit length

    # compute eigenvalue using Rayleigh quotient
    eigval = (v.T @ A @ v) / (v.T @ v)
    return eigval, v


def jacobi(A, b, iters=50):
    """
    Vectorized Jacobi method.
    A : [n,n]
    b : [n]
    """
    D = np.diag(A)                 # diagonal elements
    R = A - np.diagflat(D)         # remainder matrix

    x = np.zeros_like(b, dtype=float)

    for _ in range(iters):
        x = (b - R @ x) / D        # element-wise division

    return x

def jaccard_index(y_true, y_pred, eps=1e-8):
    """Compute Jaccard Index (IoU) for binary predictions."""
    inter = ((y_true == 1) & (y_pred == 1)).sum()  # intersection
    union = ((y_true == 1) | (y_pred == 1)).sum()  # union
    return (inter + eps) / (union + eps)  # eps prevents division by zero

def k_means(points, k, initial_centroids, max_iterations):
    """
    K-Means clustering algorithm.
    
    Args:
        points: list of tuples/lists representing data points
        k: number of clusters
        initial_centroids: initial cluster centers
        max_iterations: max number of iterations
    
    Returns:
        list of tuples: final centroid positions (rounded to 4 decimals)
    """
    # convert to tensors
    X = torch.tensor(points, dtype=torch.float32)              # [N, D]
    centroids = torch.tensor(initial_centroids, dtype=torch.float32)  # [k, D]

    for _ in range(max_iterations):

        # ---- compute distances using broadcasting ----
        # expand dims: X -> [N,1,D], centroids -> [1,k,D]
        # result: dist[i,j] = distance from point i to centroid j
        dist = torch.sum((X[:,None,:] - centroids[None,:,:])**2, dim=2)

        # ---- assign each point to nearest centroid ----
        labels = torch.argmin(dist, dim=1)   # [N]

        # ---- recompute centroids as mean of assigned points ----
        new_centroids = []
        for i in range(k):
            pts = X[labels == i]
            if len(pts) == 0:                # handle empty cluster
                new_centroids.append(centroids[i])  # keep old centroid
            else:
                new_centroids.append(pts.mean(dim=0))  # compute mean

        centroids = torch.stack(new_centroids)

    # round to 4 decimals and return as list of tuples
    return [tuple(round(float(x), 4) for x in c) for c in centroids]



# ==================== TESTS ====================

def test_mat_vec_dot():
    """Test matrix-vector dot product against PyTorch."""
    M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    v = [2, 3, 4]
    
    # Custom implementation
    result = mat_vec_dot(M, v)
    
    # PyTorch reference
    M_torch = torch.tensor(M, dtype=torch.float32)
    v_torch = torch.tensor(v, dtype=torch.float32)
    expected = (M_torch @ v_torch).tolist()
    
    # Compare
    result_np = np.array(result)
    expected_np = np.array(expected)
    diff = np.abs(result_np - expected_np).max()
    
    print(f"Matrix-Vector Dot - Custom: {result}")
    print(f"Matrix-Vector Dot - PyTorch: {expected}")
    print(f"Max difference: {diff:.6f}")
    assert diff < 1e-5, f"Expected {expected}, got {result}"

def test_matmul():
    """Test matrix multiplication against PyTorch."""
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    
    # Custom implementation
    result = matmul(A, B)
    
    # PyTorch reference
    A_torch = torch.tensor(A, dtype=torch.float32)
    B_torch = torch.tensor(B, dtype=torch.float32)
    expected = (A_torch @ B_torch).tolist()
    
    # Compare
    result_np = np.array(result)
    expected_np = np.array(expected)
    diff = np.abs(result_np - expected_np).max()
    
    print(f"Matrix Multiply - Custom: {result}")
    print(f"Matrix Multiply - PyTorch: {expected}")
    print(f"Max difference: {diff:.6f}")
    assert diff < 1e-5, f"Expected {expected}, got {result}"

def test_scalar_multiply():
    """Test scalar multiplication against PyTorch."""
    A = [[1, 2, 3], [4, 5, 6]]
    k = 3
    
    # Custom implementation
    result = scalar_multiply(A, k)
    
    # PyTorch reference
    A_torch = torch.tensor(A, dtype=torch.float32)
    expected = (k * A_torch).tolist()
    
    # Compare
    result_np = np.array(result)
    expected_np = np.array(expected)
    diff = np.abs(result_np - expected_np).max()
    
    print(f"Scalar Multiply - Max difference from PyTorch: {diff:.6f}")
    assert diff < 1e-5, f"Expected {expected}, got {result}"

def test_mean_matrix():
    """Test matrix mean against PyTorch."""
    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    
    # Custom implementation
    row_means = mean_matrix(A, "row")
    col_means = mean_matrix(A, "col")
    
    # PyTorch reference
    A_torch = torch.tensor(A, dtype=torch.float32)
    row_means_torch = A_torch.mean(dim=1).tolist()
    col_means_torch = A_torch.mean(dim=0).tolist()
    
    # Compare row means
    diff_row = np.abs(np.array(row_means) - np.array(row_means_torch)).max()
    diff_col = np.abs(np.array(col_means) - np.array(col_means_torch)).max()
    
    print(f"Mean Matrix (row) - Custom: {row_means}")
    print(f"Mean Matrix (row) - PyTorch: {row_means_torch}")
    print(f"Max difference (row): {diff_row:.6f}")
    print(f"Mean Matrix (col) - Custom: {col_means}")
    print(f"Mean Matrix (col) - PyTorch: {col_means_torch}")
    print(f"Max difference (col): {diff_col:.6f}")
    
    assert diff_row < 1e-5, f"Row means mismatch, max diff: {diff_row}"
    assert diff_col < 1e-5, f"Col means mismatch, max diff: {diff_col}"

def test_inverse_2x2():
    """Test 2x2 matrix inverse against PyTorch."""
    A = [[4, 7], [2, 6]]
    
    # Custom implementation
    inv_custom = inverse_2x2(A)
    
    # PyTorch reference
    A_torch = torch.tensor(A, dtype=torch.float32)
    inv_torch = torch.linalg.inv(A_torch).tolist()
    
    # Compare
    inv_custom_np = np.array(inv_custom)
    inv_torch_np = np.array(inv_torch)
    diff = np.abs(inv_custom_np - inv_torch_np).max()
    
    print(f"2x2 Inverse - Custom: {inv_custom}")
    print(f"2x2 Inverse - PyTorch: {inv_torch}")
    print(f"Max difference: {diff:.6f}")
    assert diff < 1e-5, f"Inverse mismatch, max diff: {diff}"


def test_covariance_matrix():
    """Test covariance matrix against NumPy."""
    np.random.seed(42)
    X_np = np.random.randn(100, 3)
    
    # Custom implementation - FIX: Pass numpy array, not torch tensor
    cov_custom = covariance_matrix(X_np)
    
    # NumPy reference
    cov_numpy = np.cov(X_np, rowvar=False)
    
    # Compare - FIX: Both are numpy arrays now
    diff = np.abs(cov_custom - cov_numpy).max()
    print(f"Covariance Matrix - Max difference from NumPy: {diff:.6f}")
    assert diff < 1e-5, f"Expected close match to NumPy, max diff: {diff}"

def test_pca():
    """Test PCA output shape and variance preservation."""
    np.random.seed(42)
    X = np.random.randn(100, 5)  # FIX: Use numpy array, not torch
    
    # Custom implementation
    X_pca = pca(X, k=2)
    
    # Check shape
    assert X_pca.shape == (100, 2), f"Expected shape (100, 2), got {X_pca.shape}"
    
    # Check that variance is preserved (principal components should be uncorrelated)
    cov_pca = np.cov(X_pca, rowvar=False)  # FIX: Use np.cov, not torch.cov
    off_diag = abs(cov_pca[0, 1])
    
    print(f"PCA - Shape: {X_pca.shape}, Off-diagonal covariance: {off_diag:.6f}")
    assert off_diag < 1e-5, f"Components should be uncorrelated, got {off_diag}"



def test_jaccard_index():
    """Test Jaccard Index against manual calculation."""
    y_true = torch.tensor([1, 0, 1, 1, 0])
    y_pred = torch.tensor([1, 1, 1, 0, 0])
    
    # Custom implementation
    result = jaccard_index(y_true, y_pred)
    
    # Manual calculation: intersection = 2, union = 4, IoU = 2/4 = 0.5
    expected = 2.0 / 4.0
    
    print(f"Jaccard Index - Custom: {result:.4f}, Expected: {expected:.4f}")
    assert abs(result - expected) < 1e-4, f"Expected {expected}, got {result}"



def test_power_method():
    """Test power method against NumPy eig."""
    np.random.seed(42)
    A = np.array([[4., 1.], [2., 3.]], dtype=np.float32)  # FIX: Use numpy array
    
    # Custom implementation
    val_custom, vec_custom = power_method(A, iters=1000)
    
    # NumPy reference (get largest eigenvalue) - FIX: Use numpy instead of torch
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = np.argmax(eigvals)
    val_numpy = eigvals[max_idx]
    vec_numpy = eigvecs[:, max_idx]
    
    # Normalize both vectors to compare (eigenvectors can differ by sign)
    vec_custom_norm = vec_custom / np.linalg.norm(vec_custom)
    vec_numpy_norm = vec_numpy / np.linalg.norm(vec_numpy)
    
    # Check if vectors are same or opposite direction
    dot_product = abs(np.dot(vec_custom_norm, vec_numpy_norm))
    
    print(f"Power Method - Custom eigenvalue: {val_custom:.6f}")
    print(f"Power Method - NumPy eigenvalue: {val_numpy:.6f}")
    print(f"Eigenvalue difference: {abs(val_custom - val_numpy):.6f}")
    print(f"Eigenvector dot product (abs): {dot_product:.6f}")
    
    assert abs(val_custom - val_numpy) < 1e-3, f"Eigenvalue mismatch"
    assert dot_product > 0.999, f"Eigenvector direction mismatch"


def test_k_means():
    """Test K-Means clustering with sklearn-like verification."""
    torch.manual_seed(42)
    
    # Create well-separated clusters
    points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
    initial = [(1, 2), (10, 2)]
    
    # Custom implementation
    result = k_means(points, 2, initial, 10)
    
    # Verify centroids are close to expected cluster centers
    # Cluster 1: mean of [(1,2), (1,4), (1,0)] = (1, 2)
    # Cluster 2: mean of [(10,2), (10,4), (10,0)] = (10, 2)
    expected_centroids = [(1.0, 2.0), (10.0, 2.0)]
    
    # Sort both by first coordinate for comparison
    result_sorted = sorted(result, key=lambda x: x[0])
    expected_sorted = sorted(expected_centroids, key=lambda x: x[0])
    
    print(f"K-Means - Custom centroids: {result_sorted}")
    print(f"K-Means - Expected centroids: {expected_sorted}")
    
    # Compare centroids
    for i in range(2):
        diff = np.sqrt((result_sorted[i][0] - expected_sorted[i][0])**2 + 
                       (result_sorted[i][1] - expected_sorted[i][1])**2)
        print(f"Centroid {i} distance: {diff:.6f}")
        assert diff < 0.1, f"Centroid {i} mismatch, distance: {diff}"


if __name__ == "__main__":
    print("=" * 60)
    print("Running tests with PyTorch/NumPy comparisons...\n")
    print("=" * 60)
    
    test_jaccard_index()
    print()
    
    test_pca()
    print()
    
    test_covariance_matrix()
    print()
    
    test_matmul()
    print()
    
    test_inverse_2x2()
    print()
    
    test_scalar_multiply()
    print()
    
    test_mat_vec_dot()
    print()
    
    test_mean_matrix()
    print()
    
    test_power_method()
    print()
    
    test_k_means()
    print()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
