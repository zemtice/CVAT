# Supplementary Notes: Why Normalization Matters in F Matrix Estimation

> **Note:** This handout covers material beyond the lecture slides.
> The slides show that using pseudo-inverse to solve for **F** produces unstable results (e.g., values like `1.251e+13`), and recommends SVD instead. This handout explains **why** instability happens and introduces **Hartley normalization** — a simple preprocessing step that makes the estimation dramatically more robust, regardless of which solver you use.
>
> The lecture slides briefly introduce an alternative formulation **Af = −1** (fixing F₃₃ = 1 and solving for 8 unknowns with exactly 8 points). We intentionally skip this formulation and focus exclusively on the homogeneous formulation **Af = 0** solved via SVD, because it generalizes to any number of points (≥ 8), makes no assumptions about F₃₃, and is the standard approach used in practice. The lecture slides themselves switch back to **Af = 0** for all subsequent examples — so you are not missing anything by focusing on this formulation only.

---

## 1. The Problem: Scale Mismatch in the Data Matrix

Recall that estimating **F** from point correspondences reduces to solving a linear system:

```
A · f = 0
```

where each row of **A** comes from one point pair (x, x') = ([u, v, 1], [u', v', 1]):

```
row = [ u'u,  u'v,  u',  v'u,  v'v,  v',  u,  v,  1 ]
```

Now consider what happens with typical image coordinates. If your image is 1920×1080, pixel coordinates range from 0 to ~1920. Look at the scale differences in a single row:

| Term | Typical magnitude |
|---|---|
| u'u | ~1920 × 1920 = **3,686,400** |
| u'v | ~1920 × 1080 = **2,073,600** |
| u', v' | ~**1000** |
| u, v | ~**1000** |
| 1 | **1** |

The entries in matrix **A** span **6 orders of magnitude** within a single row. This makes **A** extremely **ill-conditioned** — small perturbations (noise, rounding errors) in the large terms completely overwhelm the information in the small terms. SVD can partially handle this, but even SVD struggles when the condition number is this large.

> **Condition number:** A measure of how sensitive a matrix equation is to numerical errors. A well-conditioned matrix has condition number close to 1. The unnormalized **A** from image coordinates can have condition numbers in the millions.

---

## 2. The Solution: Hartley Normalization (1997)

Richard Hartley showed that a simple coordinate transformation applied **before** building matrix **A** brings all entries to the same order of magnitude, making the system well-conditioned.

The transformation has two steps, applied **independently** to each image:

### Step 1 — Translate: Move the centroid to the origin

Compute the mean position of all points, then subtract it:

```
cₓ = mean(u),   cᵧ = mean(v)

ũ = u - cₓ
ṽ = v - cᵧ
```

After this, the point cloud is centered at (0, 0).

### Step 2 — Scale: Make the average distance from origin equal to √2

Compute the average distance of all translated points from the origin, then scale so that average distance becomes √2:

```
d_mean = mean( sqrt(ũ² + ṽ²) )

s = √2 / d_mean

û = s · ũ
v̂ = s · ṽ
```

The value √2 is chosen so that a "typical" point sits at approximately (1, 1) — meaning the unit distance along each axis is 1, which balances the matrix entries perfectly.

### Matrix Form

Both steps combined can be written as a single 3×3 matrix **T**:

```
T = | s    0   -s·cₓ |
    | 0    s   -s·cᵧ |
    | 0    0    1    |
```

So the normalized point is: **x̂ = T · x** (in homogeneous coordinates).

For image 2, compute a separate **T'** using the statistics of image 2's points.

---

## 3. Effect on the Matrix Entries

After normalization, all point coordinates are roughly in the range [-2, +2]. Let's revisit the row magnitudes:

| Term | Before normalization | After normalization |
|---|---|---|
| u'u | ~3,686,400 | ~**4** |
| u'v | ~2,073,600 | ~**4** |
| u', v' | ~1000 | ~**2** |
| u, v | ~1000 | ~**2** |
| 1 | 1 | **1** |

All entries are now within the same order of magnitude. The condition number of **A** drops dramatically, and the SVD solution becomes numerically reliable.

---

## 4. The Complete Normalized 8-Point Algorithm

```
Input:  n ≥ 8 point correspondences in pixel coordinates { (xᵢ, xᵢ') }
Output: Fundamental matrix F

1. Compute normalization matrix T  from { xᵢ  }
2. Compute normalization matrix T' from { xᵢ' }

3. Normalize:   x̂ᵢ  = T  · xᵢ
                x̂ᵢ' = T' · xᵢ'

4. Build matrix A using normalized coordinates x̂ᵢ, x̂ᵢ'

5. Solve A·f = 0 via SVD:
   [U, S, V] = svd(A)
   f = last column of V  →  reshape to 3×3 to get F̂

6. Enforce rank-2 constraint on F̂:
   [U, S, V] = svd(F̂)
   S[2,2] = 0          (set smallest singular value to zero)
   F̂ = U · diag(S) · Vᵀ

7. De-normalize:
   F = T'ᵀ · F̂ · T
```

> ⚠️ **Step 7 is the most commonly forgotten step.** If you skip de-normalization, **F** lives in the normalized coordinate space and will give wrong results when applied to original pixel coordinates.

---

## 5. Before vs. After: A Concrete Comparison

The lecture slides demonstrate this difference directly. Using the **parallel configuration** example with 11 point pairs:

**Without normalization (pseudo-inverse):**
```
F = |  0.0000002   -0.000005    0.002325  |
    |  0.0000047   -0.0000022  -1.251e+13 |
    | -0.0025685    1.251e+13   1.0       |
```
The value `1.251e+13` indicates severe numerical instability.

**With normalization (SVD on normalized points):**
```
F = |  9.197e-08   -0.0000149   0.0102052 |
    |  0.0000145   -2.048e-08  -0.0090096 |
    | -0.0107188    0.0082457   1.0       |
```
All entries are in a reasonable range, and the epipolar lines computed from this **F** correctly pass through the corresponding points.

---

## 6. Summary

| | Without normalization | With Hartley normalization |
|---|---|---|
| Coordinate range | 0 ~ 1920 | approximately −2 ~ +2 |
| Matrix condition | Very large (millions) | Close to 1 |
| Numerical stability | Poor — may produce values like 1e+13 | Good |
| Accuracy of F | Unreliable | Reliable |
| Extra cost | — | Minimal (just matrix multiplications) |

Normalization adds almost no computational cost but makes the entire estimation dramatically more robust. In practice, always normalize before running the 8-point algorithm.

---

*This is supplementary material not covered in the lecture slides. For the original proof and derivation, refer to: R. Hartley, "In Defense of the Eight-Point Algorithm," IEEE TPAMI, 1997.*
