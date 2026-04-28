# From 2D–3D Correspondences to Camera Pose
## A Step-by-Step Pipeline

---

## Overview

Given at least **6 pairs of 3D–2D correspondences** $\{\mathbf{X}_i \leftrightarrow \mathbf{x}_i\}$ and the intrinsic matrix $K$, this pipeline recovers the camera's extrinsic parameters $[R \mid T]$.

```
Raw 3D/2D points
      │
      ▼  Step 1: Normalize X and x
Normalized points
      │
      ▼  Step 2: Solve P via DLT, then denormalize
Projection matrix P
      │
      ▼  Step 3: Recover RT = K⁻¹ P
Raw [R | T]
      │
      ▼  Step 4: Normalize T (remove scale ambiguity)
Scale-corrected [R | T]
      │
      ▼  Step 5: Orthogonalize R via SVD
R ∈ SO(3),  T (final)
```

---

## Step 1 — Normalize 3D Points $\mathbf{X}$ and 2D Points $\mathbf{x}$

Before running DLT, both the 3D world points and the 2D image points must be **normalized independently** (Hartley normalization). This dramatically improves the numerical stability of the linear solve.

### Why normalize?

The DLT constructs a linear system $A\mathbf{p} = 0$. When coordinates span very different magnitudes (e.g., pixel values in the hundreds vs. depth values in the thousands), the resulting matrix $A$ becomes ill-conditioned, and SVD produces an unreliable solution.

### 2D Normalization (image points)

Translate and scale so that:
- The **centroid** of the points is at the origin
- The **average distance** to the origin is $\sqrt{2}$

$$\mu_x = \frac{1}{n}\sum x_i, \quad \mu_y = \frac{1}{n}\sum y_i$$

$$\sigma = \frac{1}{n}\sum_i \sqrt{(x_i - \mu_x)^2 + (y_i - \mu_y)^2}, \quad s = \frac{\sqrt{2}}{\sigma}$$

$$[\mathbf{s}|\mathbf{t}] = \begin{bmatrix} s & 0 & -s\mu_x \\ 0 & s & -s\mu_y \\ 0 & 0 & 1 \end{bmatrix}$$

$$\tilde{\mathbf{x}}_i = [\mathbf{s}|\mathbf{t}]\,\mathbf{x}_i$$

### 3D Normalization (world points)

Same idea, but the target average distance is $\sqrt{3}$:

$$s = \frac{\sqrt{3}}{\sigma_{3D}}$$

$$[\mathbf{S}|\mathbf{T}] = \begin{bmatrix} s & 0 & 0 & -s\mu_X \\ 0 & s & 0 & -s\mu_Y \\ 0 & 0 & s & -s\mu_Z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

$$\tilde{\mathbf{X}}_i = [\mathbf{S}|\mathbf{T}]\,\mathbf{X}_i$$

> **Note:** 
> 1. Use **mean + std**  normalization (Hartley), not min–max normalization. Min–max is sensitive to outliers and is not the standard.
> 2. Normalizing to $\sqrt{d}$  ($\sqrt2$ for 2D point and $\sqrt3$ for 3D points) ​ instead of 1 ensures that the average point lies at unit distance in each individual dimension, keeping all columns of the DLT matrix $A$ at a comparable scale.

## Step 2 — Solve DLT on normalized points, then de-normalize

We have already know how to solve $P$ with at least 6 corresponding 3D ↔ 2D point pairs (Check L7 at page 10). Now solve for $\tilde{P}$ using the normalized correspondences $\{\tilde{\mathbf{X}}_i \leftrightarrow \tilde{\mathbf{x}}_i\}$, then recover the original $P$:

$$P = [\mathbf{s}|\mathbf{t}]^{-1}\,\tilde{P}\,[\mathbf{S}|\mathbf{T}]$$

---

## Step 3 — Recover $[R \mid T]$ from $K$ and $P$

Given the known intrinsic matrix $K$ and the recovered projection matrix $P$, we use:

$$P = K\begin{bmatrix}R & T\end{bmatrix}$$

Left-multiply both sides by $K^{-1}$:

$$K^{-1}P = \begin{bmatrix}R & T\end{bmatrix}$$

```python
K_inv = np.linalg.inv(K)
RT = K_inv @ P          # shape: (3, 4)

R_raw = RT[:, :3]       # first three columns
t_raw = RT[:, 3]        # last column
```

> At this point, `R_raw` is **not yet** a valid rotation matrix and `t_raw` carries an unknown scale factor, both due to the scale ambiguity inherent in $P$.

---

## Step 4 — Normalize $T$ (Remove Scale Ambiguity)

$P$ is defined only **up to a global scale**: $\lambda P$ and $P$ describe the same camera. Therefore $K^{-1}P$ also carries an unknown factor $\lambda$.

A valid rotation matrix must have **unit-norm columns**:

$$\|\mathbf{r}_1\| = \|\mathbf{r}_2\| = \|\mathbf{r}_3\| = 1$$

We use the norm of the **first column of $R$** as the scale estimate and divide the entire $[R \mid T]$ by it:

$$\lambda = \|\mathbf{r}_1\|, \quad \begin{bmatrix}R & T\end{bmatrix} \leftarrow \frac{1}{\lambda}\begin{bmatrix}R_{\text{raw}} & \mathbf{t}_{\text{raw}}\end{bmatrix}$$

```python
scale = np.linalg.norm(R_raw[:, 0]) # norm of first colum
RT_normalized = RT / scale

R_scaled = RT_normalized[:, :3]
T         = RT_normalized[:, 3:]   # shape: (3, 1)
```

After this step, $R_{\text{scaled}}$ has unit-norm columns, and $T$ is in the correct physical scale.

---

## Step 5 — Orthogonalize $R$ via SVD

Even after scale normalization, numerical errors (especially when $P$ was estimated from noisy data via DLT) can cause $R_{\text{scaled}}$ to slightly violate the orthogonality constraint $R^\top R = I$.

We project $R$ onto $SO(3)$ using SVD:

$$R_{\text{scaled}} = U \Sigma V^\top$$

$$R = UV^\top$$

If $\det(UV^\top) = -1$ (a reflection, not a rotation), flip the sign of the last column of $U$:

$$U[:, -1] \;\mathrel{*}=\; -1, \quad R = UV^\top$$

```python
U, _, Vt = np.linalg.svd(R_scaled)
R = U @ Vt

# Ensure det(R) = +1 (proper rotation, not reflection)
if np.linalg.det(R) < 0:
    U[:, -1] *= -1
    R = U @ Vt
```
