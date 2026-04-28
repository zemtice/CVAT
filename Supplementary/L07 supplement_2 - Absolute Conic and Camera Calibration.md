# Absolute Conic and Camera Calibration

> Course: Applications of Computer Vision Algorithms | Topic: Camera Calibration

---

## 1. Definition of Invariance

In geometry, we say a **geometric object** is **invariant** under a **transformation** if the following holds:

> Given a set of points $S$ and a transformation $H$. If for every point $\mathbf{p} \in S$, the transformed point $H\mathbf{p}$ still belongs to $S$; and no point outside $S$ is mapped into $S$, then $S$ is said to be **invariant** under $H$.

In mathematical notation:

$$\mathbf{p} \in S \iff H\mathbf{p} \in S$$

Note that this is an **if and only if ($\iff$)** condition, not a one-way implication ($\Rightarrow$). Both directions must hold to guarantee that the set is identical before and after the transformation.

---

## 2. Hierarchy of Geometric Transformations

The three transformations below form a hierarchy — each level permits more operations than the one above it.

### Euclidean Transformation

Also known as **rigid body motion**, allowing only rotation and translation:

$$\mathbf{X}' = \begin{pmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix} \mathbf{X}, \quad R \in SO(3),\; \mathbf{t} \in \mathbb{R}^3$$

- $R^\top R = I$, $\det R = 1$

**Preserved quantities**: distances, angles, areas, volumes

### Similarity Transformation

Extends the Euclidean transformation by allowing **uniform scaling**:

$$\mathbf{X}' = \begin{pmatrix} sR & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix} \mathbf{X}, \quad s > 0$$

**Preserved quantities**: angles, **ratios** of distances

### Affine Transformation

Allows any invertible linear map plus translation, including non-uniform scaling and shear:

$$\mathbf{X}' = \begin{pmatrix} A & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix} \mathbf{X}$$

- $A$: any invertible $3 \times 3$ matrix (not required to be orthogonal)

**Preserved quantities**: parallelism, area ratios (angles are generally not preserved)

### Summary

| Transformation | Rotation | Translation | Uniform Scaling | Non-uniform Scaling / Shear |
|----------------|:--------:|:-----------:|:---------------:|:---------------------------:|
| Euclidean      | ✓ | ✓ | ✗ | ✗ |
| Similarity     | ✓ | ✓ | ✓ | ✗ |
| Affine         | ✓ | ✓ | ✓ | ✓ |

---

## 3. Camera Transformations: Extrinsics + Intrinsics

The full pipeline from a 3D scene point $\mathbf{X}_w$ to an image coordinate $\mathbf{x}$ can be decomposed into two stages:

$$\mathbf{x} = K \underbrace{[R \mid \mathbf{t}]}_{\text{extrinsics}} \mathbf{X}_w$$

### Extrinsic Parameters: Euclidean Transformation

$[R \mid \mathbf{t}]$ describes the transformation from the **world coordinate system to the camera coordinate system** — a Euclidean transformation (rotation + translation):

$$\mathbf{X}_c = R\mathbf{X}_w + \mathbf{t}$$

This step preserves all geometric relationships; distances and angles remain unchanged.

### Intrinsic Parameters: Affine Transformation

The intrinsic matrix $K$ maps from **camera coordinates to pixel coordinates**:

$$K = \begin{pmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$$

- $f_x, f_y$: focal lengths along x and y (in pixels) — allow **non-uniform scaling**
- $s$: skew coefficient due to non-perpendicular pixel axes (usually $s \approx 0$ in modern cameras)
- $c_x, c_y$: principal point offset (**translation**)

Because $K$ includes non-uniform scaling and skew, it is fundamentally an **affine transformation**, not a similarity transformation.

### Key Observation

> The Euclidean transformation (extrinsics) preserves all geometric structure. The **affine transformation (intrinsics) distorts angles and distance ratios**. The goal of camera calibration is to recover this distortion parameter $K$.

---

## 4. The Absolute Conic: Bridging Geometric Invariance and Camera Calibration

### Core Idea

We are looking for a 3D geometric object that satisfies:

1. **Invariant under Euclidean/similarity transformations** — extrinsic parameters cannot alter it
2. **Altered by the affine transformation (intrinsics)** — its projected appearance has a fixed mathematical relationship with $K$

If such an object exists, then observing its "distorted image" allows us to recover $K$.

The **Absolute Conic $\Omega_\infty$** possesses exactly these two properties.

### Definition

The Absolute Conic is a conic curve defined on the **plane at infinity $\pi_\infty$** (where $X_4 = 0$):

$$\Omega_\infty = \left\{ (X_1 : X_2 : X_3 : 0)^\top \;\middle|\; X_1^2 + X_2^2 + X_3^2 = 0 \right\}$$

Two defining conditions:

- **Condition A**: $X_4 = 0$ (the point lies on the plane at infinity)
- **Condition B**: $X_1^2 + X_2^2 + X_3^2 = 0$ (the direction vector has zero complex norm)

> Condition B has no real-valued solution, so all points on $\Omega_\infty$ are **complex points**. Its significance comes from its algebraic structure, not from any visually observable shape.

#### Point on $\Omega_\infty$ example:

| Point               | On $\pi_\infty$? |    On $\Omega_\infty$?     |
| ------------------- | :--------------: | :------------------------: |
| $[1, i, 0, 0]^\top$ |        ✅         |    ✅ ($1 - 1 + 0 = 0$)     |
| $[1, 0, 0, 0]^\top$ |        ✅         | ❌ ($1 + 0 + 0 = 1 \neq 0$) |


---

## 5. Why Does the Absolute Conic Have Euclidean/Similarity Invariance?

We verify the invariance condition from Section 1:

$$\mathbf{p} \in \Omega_\infty \iff H\mathbf{p} \in \Omega_\infty$$

We decompose the proof into the two defining conditions.

Let $\mathbf{p} = (X_1, X_2, X_3, 0)^\top = \begin{pmatrix} \mathbf{d} \\ 0 \end{pmatrix}$ be a point on $\Omega_\infty$. Applying a Euclidean or similarity transformation:

$$H\mathbf{p} = \begin{pmatrix} sR & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix} \begin{pmatrix} \mathbf{d} \\ 0 \end{pmatrix} = \begin{pmatrix} sR\mathbf{d} \\ 0 \end{pmatrix}$$

### Condition A: $\pi_\infty$ is unchanged

The last component remains $0$, because the translation term $\mathbf{t}$ is multiplied by $X_4 = 0$ and vanishes. Therefore:

$$\mathbf{p} \in \pi_\infty \iff H\mathbf{p} \in \pi_\infty \quad \checkmark$$

> Note: affine transformations also preserve $\pi_\infty$, but general **projective transformations do not** — they can map points at infinity to finite points.

### Condition B: The equation is preserved under rotation and uniform scaling

With $\mathbf{d}' = sR\mathbf{d}$, substituting into Condition B:

$$\|\mathbf{d}'\|^2 = (sR\mathbf{d})^\top(sR\mathbf{d}) = s^2\,\mathbf{d}^\top \underbrace{R^\top R}_{=\,I} \mathbf{d} = s^2\|\mathbf{d}\|^2$$

Since $s^2 > 0$, the factor can be cancelled in homogeneous coordinates, giving an equivalence in both directions:

$$\|\mathbf{d}'\|^2 = 0 \iff \|\mathbf{d}\|^2 = 0 \quad \checkmark$$

### Why does an affine transformation break Condition B?

Under an affine transformation $\mathbf{d}' = A\mathbf{d}$:

$$\|\mathbf{d}'\|^2 = \mathbf{d}^\top A^\top A\,\mathbf{d}$$

Since $A^\top A \neq s^2 I$ in general, the set of points satisfying $\|\mathbf{d}'\|^2 = 0$ is no longer the same as the set satisfying $\|\mathbf{d}\|^2 = 0$. This is precisely why $K$ (an affine transformation) maps $\Omega_\infty$ to a different conic $\omega$ in the image.

### Complete Logic Chain

$$\mathbf{p} \in \Omega_\infty$$
$$\iff \underbrace{X_4 = 0}_{\text{Condition A}} \;\text{ and }\; \underbrace{\|\mathbf{d}\|^2 = 0}_{\text{Condition B}}$$
$$\iff \underbrace{(H\mathbf{p})_4 = 0}_{\text{Condition A preserved}} \;\text{ and }\; \underbrace{s^2\|\mathbf{d}\|^2 = 0 \iff \|\mathbf{d}\|^2 = 0}_{\text{Condition B preserved ($s^2 > 0$ cancels)}}$$
$$\iff H\mathbf{p} \in \Omega_\infty \quad \blacksquare$$
In summary, we have 
### Role in the Camera Projection Pipeline

| Stage                            | Transformation Type | Effect on $\Omega_\infty$                                       |
| -------------------------------- | ------------------- | --------------------------------------------------------------- |
| Extrinsics $[R \mid \mathbf{t}]$ | Euclidean           | **Unchanged**: $\Omega_\infty$ is invariant                     |
| Intrinsics $K$                   | Affine              | **Changed**: $\Omega_\infty$ is mapped to $\omega$ in the image |
Because the extrinsics are "transparent" to $\Omega_\infty$, the observed $\omega$ in the image depends only on $K$, making it a direct tool for intrinsic parameter estimation.

---

## 6. Image of the Absolute Conic (IAC) and $\omega$

### Derivation of $\omega = (KK^\top)^{-1}$

**Step 1**: A point $[\mathbf{d}^{\top}, 0]^{\top}$ on $\Omega_\infty$ projects to image point $\mathbf{x}$ (translation vanishes since $X_4 = 0$):

$$\mathbf{x} = KR\mathbf{d} \implies \mathbf{d} = R^{-1}K^{-1}\mathbf{x}$$

**Step 2**: Substitute into Condition B ($\mathbf{d}^\top\mathbf{d} = 0$):

$$\mathbf{x}^\top \underbrace{K^{-\top}R^{-\top}R^{-1}K^{-1}}_{\omega} \mathbf{x} = 0$$
$\omega$ is actually a conic in 2D image plane in homogeneous coordinate (remember in Lecture 3, conic $C$ can be represent by a $3\times3$ matrix, the point $\mathbf{x}$ on conic will satisfy $\mathbf{x}^{\top}C\mathbf{x}=0$).

**Step 3**: Since $R^{-1} = R^\top$, we have $R^{-\top}R^{-1} = I$, giving:

$$\boxed{\omega = K^{-\top}K^{-1} = (KK^\top)^{-1}}$$

$\omega$ is the "shadow" of $\Omega_\infty$ after being distorted by $K$ onto the image plane. Since the extrinsics are transparent to $\Omega_\infty$, $\omega$ depends only on $K$.

---

## 7. The Orthogonality Condition: $\mathbf{l}^\top\omega\mathbf{m} = 0$

If two image directions $\mathbf{l}$ and $\mathbf{m}$ correspond to mutually perpendicular 3D directions, then:

$$\mathbf{l}^\top \omega \mathbf{m} = 0$$

### Derivation

The two 3D directions are $\mathbf{d}_1 = R^{-1}K^{-1}\mathbf{l}$ and $\mathbf{d}_2 = R^{-1}K^{-1}\mathbf{m}$. Substituting $\mathbf{d}_1^\top\mathbf{d}_2 = 0$:

$$\mathbf{l}^\top K^{-\top} \underbrace{R^{-\top}R^{-1}}_{=I} K^{-1}\mathbf{m} = \mathbf{l}^\top \underbrace{(KK^\top)^{-1}}_{=\omega} \mathbf{m} = 0$$

**Intuition**: $\omega$ acts as a "restorer" — it undoes the distortion introduced by $K$, allowing us to measure true 3D orthogonality directly from image observations.

---

## 8. Zhang's Method: Estimating $\omega$ in Practice

### Pipeline

```
Capture checkerboard images (multiple orientations)
        ↓
Estimate one Homography H per image
        ↓
Each H provides 2 constraint equations on ω
        ↓
Stack all constraints and solve for ω
        ↓
Cholesky decomposition → K
```

### Why Can a Homography Provide Constraints?

Set the checkerboard as the $Z = 0$ plane in world coordinates. Substituting into the projection matrix:

$$\mathbf{x} = K[R \mid \mathbf{t}]\begin{bmatrix}X\\Y\\0\\1\end{bmatrix} = \underbrace{K[\mathbf{r}_1, \mathbf{r}_2 \mid \mathbf{t}]}_{H}\begin{bmatrix}X\\Y\\1\end{bmatrix}$$

Setting $Z = 0$ eliminates $\mathbf{r}_3$, giving:

$$H = [\mathbf{h}_1,\ \mathbf{h}_2,\ \mathbf{h}_3] = [K\mathbf{r}_1,\ K\mathbf{r}_2,\ K\mathbf{t}]$$

| Column | Equals | Geometric Meaning |
|--------|--------|-------------------|
| $\mathbf{h}_1$ | $K\mathbf{r}_1$ | Checkerboard X-axis projected onto image |
| $\mathbf{h}_2$ | $K\mathbf{r}_2$ | Checkerboard Y-axis projected onto image |
| $\mathbf{h}_3$ | $K\mathbf{t}$   | Translation component |

### Two Constraints per Image

Using $\mathbf{r}_1 \perp \mathbf{r}_2$ and $\|\mathbf{r}_1\| = \|\mathbf{r}_2\| = 1$, together with the orthogonality condition from Section 7:

$$\mathbf{h}_1^\top\omega\mathbf{h}_2 = 0 \quad \text{(X and Y directions are perpendicular)}$$
$$\mathbf{h}_1^\top\omega\mathbf{h}_1 = \mathbf{h}_2^\top\omega\mathbf{h}_2 \quad \text{(X and Y directions have equal norm)}$$

Substituting $\mathbf{h}_i = K\mathbf{r}_i$ and $\omega = (KK^\top)^{-1}$ verifies:

$$\mathbf{h}_i^\top\omega\mathbf{h}_j = \mathbf{r}_i^\top\mathbf{r}_j$$

These two constraints are therefore equivalent to the orthonormality conditions on the rotation matrix columns — geometrically guaranteed constraints, not arbitrary assumptions.

### How Many Images Are Needed?

$\omega$ is a $3 \times 3$ symmetric matrix; removing the scale ambiguity leaves **5 unknowns**. Each image contributes 2 constraints:

$$\text{Minimum: } \lceil 5/2 \rceil = 3 \text{ images}$$

In practice, **10–20 images** are recommended to improve stability via least squares.

### Recovering $K$ from $\omega$

$$\omega = (KK^\top)^{-1} \implies KK^\top = \omega^{-1}$$

Apply **Cholesky decomposition** to $\omega^{-1}$ (which is symmetric positive definite) to directly obtain the upper triangular matrix $K$.

---

## Summary

| Concept | Description |
|---------|-------------|
| Invariance | $\mathbf{p} \in S \iff H\mathbf{p} \in S$, both directions required |
| Euclidean / Similarity / Affine | Progressively relaxed; affine allows non-uniform scaling and shear |
| Camera transformations | Extrinsics = Euclidean; intrinsics $K$ = Affine |
| Role of $\Omega_\infty$ | Invariant to Euclidean/similarity → extrinsics are transparent; broken by affine → image appearance encodes $K$ |
| Source of invariance | $R^\top R = I$ neutralizes rotation; $s^2 > 0$ cancels in homogeneous coordinates; translation vanishes at infinity |
| IAC $\omega$ | Projection of $\Omega_\infty$ onto the image plane, $\omega = (KK^\top)^{-1}$ |
| Orthogonality condition | 3D perpendicular $\iff$ $\mathbf{l}^\top\omega\mathbf{m} = 0$ |
| Zhang's method | Checkerboard → Homography → 2 constraints/image → solve $\omega$ → Cholesky → $K$ |
