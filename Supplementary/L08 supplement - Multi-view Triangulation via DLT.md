# Multi-view 3D Reconstruction via Direct Linear Triangulation (DLT)

> Course: Computer Vision and Applications - Tutorials | Topic: Multi-view Geometry / 3D Reconstruction

---

## 1. The Triangulation Problem
![](https://res.cloudinary.com/dx77thjfu/image/upload/Screenshot_2026-04-28_143115_hccjqt.png)

Given $N$ calibrated cameras (each with a known $3 \times 4$ projection matrix $P_i$) that all observe the **same** 3D world point $\mathbf{X}$, and given the corresponding 2D image observations $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N$, recover the 3D coordinates of $\mathbf{X}$.

A calibrated camera maps a 3D world point $\mathbf{X} = (X, Y, Z, 1)^\top$ in homogeneous coordinates to a 2D image point $\mathbf{x} = (u, v, 1)^\top$ through:

$$\lambda\,\mathbf{x} = P\,\mathbf{X}$$

The unknown scalar $\lambda$ is the **projective depth** — it absorbs the fact that $\mathbf{x}$ is only defined up to scale in homogeneous coordinates. This depth is different for each camera and is **not something we want to solve for**, so we need a way to eliminate it.

---

## 2. Eliminating the Depth Scalar via the Cross Product

### The Key Trick

Any non-zero vector is parallel to itself, and the cross product of two parallel vectors is zero:

$$\mathbf{x} \times (P\mathbf{X}) = \mathbf{0}$$

Both sides of $\lambda\mathbf{x} = P\mathbf{X}$ point in the same direction, so their cross product vanishes regardless of $\lambda$. The unknown depth has been **algebraically eliminated**.

### The Skew-Symmetric (Cross-Product) Matrix

The cross product $\mathbf{x} \times \mathbf{y}$ can be rewritten as a matrix-vector multiplication using the **skew-symmetric matrix** $[\mathbf{x}]_\times$:

$$[\mathbf{x}]_\times \;=\; \begin{bmatrix} 0 & -1 & v \\ 1 & 0 & -u \\ -v & u & 0 \end{bmatrix} \quad\text{such that}\quad \mathbf{x} \times \mathbf{y} = [\mathbf{x}]_\times \mathbf{y}$$

So the projection constraint becomes a **linear** equation in the unknown $\mathbf{X}$:

$$[\mathbf{x}]_\times \, P \, \mathbf{X} \;=\; \mathbf{0}$$

This is the heart of DLT: a non-linear projective relation has been converted into a **homogeneous linear system** in $\mathbf{X}$.

---

## 3. Linear Constraint per Camera

The expression $[\mathbf{x}]_\times P \mathbf{X} = \mathbf{0}$ is a block of **3 scalar equations**. However, $[\mathbf{x}]_\times$ is a $3 \times 3$ skew-symmetric matrix and is therefore **rank-2** — its three rows are linearly dependent.

> Geometrically, $[\mathbf{x}]_\times \mathbf{y} = \mathbf{0}$ says "$\mathbf{y}$ is parallel to $\mathbf{x}$". That is a 1-dimensional condition (parallelism), so it can only impose 2 independent constraints on the 3 components of $\mathbf{y}$.

So **each camera contributes 2 linearly independent equations** in the 4 homogeneous unknowns of $\mathbf{X}$.

| Per-camera quantity      | Count |
| ------------------------ | :---: |
| Equations in the block   |  3   |
| Linearly independent     |  2   |
| Unknowns in $\mathbf{X}$ |  4   |

---

## 4. Stacking Constraints from N Cameras

With $N$ cameras observing the same 3D point, we stack all per-camera blocks into one large linear system:

$$\underbrace{\begin{bmatrix} [\mathbf{x}_1]_\times P_1 \\ [\mathbf{x}_2]_\times P_2 \\ \vdots \\ [\mathbf{x}_N]_\times P_N \end{bmatrix}}_{A\;:\;(3N \times 4)} \mathbf{X} \;=\; \mathbf{0}$$

For **N = 3 cameras**, $A$ is $9 \times 4$. After accounting for the rank-2 redundancy in each block, we have $2N = 6$ effective equations and 4 unknowns — the system is **overdetermined**, which is exactly what we want for noise-tolerant estimation.

| Number of cameras | $A$ shape | Effective equations | Status                           |
| :---------------: | :-------: | :-----------------: | -------------------------------- |
|       $N = 2$       |  $6 \times 4$  |       $4$         | Minimal (exactly determined)     |
|       $N \geq 3$       |  $3N \times 4$  |       $2N$         | Overdetermined — improves accuracy       |

---

## 5. Dehomogenization

Solving the homogeneous system $A\mathbf{X} = \mathbf{0}$ in the least-squares sense (via SVD, as covered in earlier lectures) returns $\mathbf{X} = (X_1, X_2, X_3, X_4)^\top$ in homogeneous coordinates. Recover Cartesian coordinates by dividing through:

$$\mathbf{X}_{\text{cartesian}} = \left( \frac{X_1}{X_4},\; \frac{X_2}{X_4},\; \frac{X_3}{X_4} \right)$$

> **Edge case:** if $X_4 \approx 0$, the recovered point is *at infinity* — usually a sign that the camera baselines are too small or the rays are nearly parallel. In practice, robust pipelines flag and discard such reconstructions.

---

## 6. Beyond the Slides: Algebraic vs Geometric Error

Triangulation linearised via the cross-product trick is called **DLT triangulation**. It is fast, closed-form, and is the standard **initialisation** for non-linear refinement. However, there is an important caveat that the slides typically gloss over:

### What DLT Actually Minimises

The DLT solution minimises $\|A\mathbf{X}\|^2 = \sum_i \|[\mathbf{x}_i]_\times P_i \mathbf{X}\|^2$.

This is an **algebraic error**: the squared norm of the residual of the linearised equations. It has **no direct geometric meaning** — it does not correspond to pixel distance, ray distance, or any quantity a practitioner would care about.

### What We Actually Want: Geometric (Reprojection) Error

The geometrically meaningful quantity is the **reprojection error**: the sum of squared pixel distances between observed image points and the projection of $\mathbf{X}$:

$$E_{\text{reproj}}(\mathbf{X}) \;=\; \sum_{i=1}^{N} \left\| \mathbf{x}_i - \pi(P_i \mathbf{X}) \right\|^2$$

where $\pi$ denotes the perspective division $(x, y, w) \mapsto (x/w, y/w)$.

This is **non-linear** in $\mathbf{X}$ (because of the division by the depth component) and cannot be solved in closed form.

### The Practical Pipeline

| Step | Method                                  | Purpose                              |
| :--: | --------------------------------------- | ------------------------------------ |
|  1   | DLT (this note)                         | Get a fast, closed-form initial guess |
|  2   | Levenberg–Marquardt non-linear refinement | Minimise true reprojection error     |

> For **clean correspondences** (e.g. a controlled lab setup, synthetic data, or this practice exercise), DLT alone is already accurate enough. For **real-world data** with noisy feature matches, the LM refinement step is non-negotiable — bundle adjustment in SfM/SLAM systems essentially performs this refinement jointly over all cameras and points.

---

## 7. Implementation Notes

### Numerical Conditioning

The entries of $A$ mix pixel coordinates (hundreds to thousands) with the entries of $P$ (which can vary widely in scale). This produces a poorly conditioned matrix. Two standard remedies:

1. **Normalise the image points** so their centroid is at the origin and their average distance from the origin is $\sqrt{2}$ (Hartley normalisation, also seen in Lecture 6 for the F-matrix).
2. **Normalise the projection matrices** consistently with the image normalisation.

Without normalisation, DLT can be unstable for cameras with very different focal lengths or image resolutions.

### Why Use Cross Product Instead of Just Dividing?

A naive alternative is to solve $\lambda\mathbf{x} = P\mathbf{X}$ by taking ratios, e.g. $u = (P_1 \mathbf{X}) / (P_3 \mathbf{X})$. This works algebraically but introduces **division by an unknown**, which is non-linear and numerically dangerous when the denominator is small. The cross product avoids this entirely — every operation is linear in $\mathbf{X}$.

---

## Summary

| Concept                        | Description                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Projection model               | $\lambda \mathbf{x} = P \mathbf{X}$ with unknown depth $\lambda$                                       |
| Cross-product trick            | $\mathbf{x} \times (P\mathbf{X}) = \mathbf{0}$ eliminates $\lambda$ algebraically                      |
| Skew-symmetric matrix          | $[\mathbf{x}]_\times$ converts cross product to matrix-vector multiplication                          |
| Per-camera contribution        | 3 equations, but rank-2 ⇒ **2 linearly independent constraints**                                      |
| Stacked system                 | $A \mathbf{X} = \mathbf{0}$, with $A$ of shape $3N \times 4$ — solve via SVD as before                |
| Dehomogenization               | Divide $(X_1, X_2, X_3, X_4)^\top$ by $X_4$                                                            |
| Error minimised                | **Algebraic** error $\|A\mathbf{X}\|^2$ — *not* the geometric reprojection error                       |
| Use in practice                | Initialise non-linear refinement (Levenberg–Marquardt / bundle adjustment) for noisy real-world data  |
| Numerical hygiene              | Normalise image points and projection matrices to avoid ill-conditioning                              |
