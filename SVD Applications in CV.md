# SVD Applications in Computer Vision

> **Topic:** Singular Value Decomposition (SVD) — Theory & Practice
> **Update:** 2026-03-24

---

## 1. SVD Review

Given a matrix $A \in \mathbb{R}^{m \times n}$, SVD decomposes it as:

$$
A = U \Sigma V^T
$$

- $U \in \mathbb{R}^{m \times m}$: left singular vectors (orthogonal)
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$: right singular vectors (orthogonal)

**Key insight:** The singular values indicate how much "information" each component carries. Large singular values = important structure; small singular values ≈ noise.

---

## 2. Solving Ax = b (Least-Squares Solution)

### Motivation

In practice, we often have more equations than unknowns (overdetermined system). An exact solution may not exist — we want the **best approximate solution** that minimizes $\|Ax - b\|$.

### Solution via SVD

# 

Using the **Moore-Penrose pseudoinverse**:

$$
x = A^+ b = V \Sigma^+ U^T b
$$

where $\Sigma^+$ replaces each non-zero singular value $\sigma_i$ with $1/\sigma_i$.

### Python Example

```python
import numpy as np

# Overdetermined system: 4 equations, 2 unknowns
# True answer: x = [2, 3]
A = np.array([[1, 1],
              [2, 1],
              [3, 1],
              [4, 1]], dtype=float)
b = np.array([5, 7, 9, 11], dtype=float)

# Method 1: np.linalg.lstsq (uses SVD internally)
x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
print("lstsq solution:", x_lstsq)

# Method 2: Manual SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)
s_inv = np.diag(1.0 / s)
A_pinv = Vt.T @ s_inv @ U.T
x_svd = A_pinv @ b
print("SVD solution:  ", x_svd)

# Verify
print("Residual ||Ax - b||:", np.linalg.norm(A @ x_svd - b))
```

**Output:**

```
lstsq solution:  [2. 3.]
SVD solution:    [2. 3.]
Residual ||Ax - b||: ~0.0
```

---

## 3. Solving Ax = 0 (Homogeneous System / DLT)

### Motivation

Many computer vision problems reduce to finding a non-trivial vector $x$ such that $Ax \approx 0$, subject to $\|x\| = 1$.

Examples:

- **Homography estimation (find H)**
- **Fundamental matrix estimation (find F)**
- **Camera calibration (find P)**

### Solution via SVD (the "SVD trick")

Apply SVD to $A$. The solution $x$ is the **last row of $V^T$** — i.e., the right singular vector corresponding to the **smallest singular value**.

This minimizes $\|Ax\|$ subject to $\|x\| = 1$.

### Python Example: Homography Estimation (DLT)

```python
import numpy as np

def build_A_matrix(pts_src, pts_dst):
    """
    Build the DLT matrix A.
    Each point correspondence contributes 2 rows.
    """
    A = []
    for (x, y), (xp, yp) in zip(pts_src, pts_dst):
        A.append([-x, -y, -1,  0,  0,  0, xp*x, xp*y, xp])
        A.append([ 0,  0,  0, -x, -y, -1, yp*x, yp*y, yp])
    return np.array(A)

def compute_homography(pts_src, pts_dst):
    """
    Compute the Homography matrix using DLT + SVD.
    Requires at least 4 point correspondences.
    """
    A = build_A_matrix(pts_src, pts_dst)

    # SVD decomposition
    U, s, Vt = np.linalg.svd(A)

    # Solution = last row of Vt (smallest singular value)
    h = Vt[-1]
    H = h.reshape(3, 3)

    # Normalize so that H[2,2] = 1
    H = H / H[2, 2]
    return H

# Test: 4 point correspondences (scale x2)
pts_src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
pts_dst = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)

H = compute_homography(pts_src, pts_dst)
print("Estimated Homography:")
print(np.round(H, 4))

# Verify: project src points through H
print("\nVerification:")
for p in pts_src:
    ph = np.array([p[0], p[1], 1.0])
    result = H @ ph
    result /= result[2]
    print(f"  {p} -> {result[:2]}")
```

**Expected output (2x scaling homography):**

```
[[2. 0. 0.]
 [0. 2. 0.]
 [0. 0. 1.]]
```

---

## 4. Summary Table

| Problem                        | Form     | SVD Role                                |
| ------------------------------ | -------- | --------------------------------------- |
| Least-squares fitting          | $Ax = b$ | Pseudoinverse $A^+ = V\Sigma^+ U^T$     |
| Homography                     | $Ax = 0$ | DLT: $x$ = last row of $V^T$            |
| Fundamental matrix             | $Ax = 0$ | Same as DLT                             |
| Camera calibration             | $Ax = 0$ | Same as DLT                             |
| PCA / Dimensionality reduction | —        | Singular vectors = principal directions |
| Image compression              | —        | Keep top-$k$ singular values            |
