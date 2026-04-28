# Supplementary Notes: Estimating Camera Pose from the Essential Matrix

> **Note:** This handout covers material beyond the lecture slides.
> It explains the *practical* pipeline for estimating the Essential matrix **E** and recovering camera pose **(R, t)** from two images — a workflow used in real-world 3D reconstruction tools such as COLMAP and ORB-SLAM.

---

## 0. Prerequisites

You should already be familiar with:
- Camera intrinsic matrix **K** (from camera calibration)
- Epipolar geometry: epipole, epipolar line, Fundamental matrix **F**
- The relationship **E = K'ᵀ F K**

---

## 1. What Does the Essential Matrix Give Us?

Given two images taken from different positions **by calibrated cameras**, the Essential matrix **E** encodes the **relative rotation R** and **translation direction t** between them.

From **E** alone you can recover:
- The rotation matrix **R** (how the camera was rotated)
- The **direction** of translation **t** (where the camera moved to)

> ⚠️ **Important limitation:** The *scale* of translation cannot be recovered from images alone. You know the camera moved in direction **t**, but not *how far*. Recovering true scale requires external information (e.g., a known object size, an IMU, or a stereo baseline).

---

## 2. The Full Pipeline

```
Two images
    ↓  Step 1: Feature detection & matching
Pixel correspondences  (x, x')
    ↓  Step 2: Convert to normalized coordinates
Normalized correspondences  (x̂, x̂')
    ↓  Step 3: Estimate E  (5-point algorithm + RANSAC)
E  (raw estimate)
    ↓  Step 4: Enforce rank-2 constraint
E  (corrected)
    ↓  Step 5: Decompose via SVD → 4 candidate solutions
(R₁,t₁)  (R₂,t₂)  (R₃,t₃)  (R₄,t₄)
    ↓  Step 6: Cheirality check
Correct  (R, t)  ← camera relative pose
```

---

## 3. Step-by-Step Explanation

### Step 1 — Feature Detection and Matching

Use a feature detector (e.g., SIFT, ORB) to find distinctive keypoints in both images, then match them across the two views to obtain a set of point correspondences:

```
{ (x₁, x₁'),  (x₂, x₂'),  ...,  (xₙ, xₙ') }
```

where **xᵢ** is a pixel in image 1 and **xᵢ'** is the corresponding pixel in image 2.

> **Practical note:** Raw matches always contain outliers (wrong matches). You must run **RANSAC** together with the estimation step (Step 3) to reject them. Skipping RANSAC will produce a completely wrong **E**.

---

### Step 2 — Convert to Normalized Image Coordinates

The Essential matrix is defined in **normalized image coordinates** (i.e., as if the camera had no intrinsics). Before estimating **E**, convert each pixel coordinate using the intrinsic matrix **K**:

```
x̂  = K⁻¹ · x       (for image 1)
x̂' = K'⁻¹ · x'     (for image 2)
```

This removes the effect of focal length, principal point, and pixel skew. The result is a coordinate in meters on the virtual "unit plane" in front of the camera.

> If the two cameras have **different** intrinsic matrices (K ≠ K'), apply the appropriate inverse to each side separately.

---

### Step 3 — Estimate E with the 5-Point Algorithm + RANSAC

**E** has **5 degrees of freedom** (3 for rotation, 2 for translation direction — scale is arbitrary). Therefore, the theoretical minimum number of point pairs needed is **5**.

The **5-point algorithm** is the standard choice in practice:
- Requires only 5 correspondences per RANSAC hypothesis → faster than the 8-point algorithm
- More numerically stable
- Built into OpenCV as `cv2.findEssentialMat()`

```python
import cv2

E, mask = cv2.findEssentialMat(
    pts1_norm,   # normalized coordinates from image 1
    pts2_norm,   # normalized coordinates from image 2
    focal=1.0,   # already normalized, so focal=1
    pp=(0., 0.), # principal point at origin after normalization
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1e-3
)
```

> **Why not use F directly?**
> You *could* estimate **F** from pixel coordinates and then compute **E = K'ᵀ F K**. But estimating **E** directly from normalized coordinates is more accurate when **K** is known, because the normalization already conditions the numerical problem well.

---

### Step 4 — Enforce the Rank-2 Constraint

A valid Essential matrix must satisfy two conditions:
1. **Rank 2** — it has exactly one zero singular value
2. **Two equal singular values** — the non-zero singular values must be equal: σ₁ = σ₂

In practice, the numerically estimated **E** will not satisfy these exactly. You must correct it via SVD:

```
E = U · diag(σ₁, σ₂, σ₃) · Vᵀ
           ↓  replace singular values
E = U · diag(1, 1, 0) · Vᵀ
```

```python
U, S, Vt = np.linalg.svd(E)
E_corrected = U @ np.diag([1, 1, 0]) @ Vt
```

> ⚠️ **Do not skip this step.** If you decompose the uncorrected **E**, the recovered **R** will not be a valid rotation matrix (det(R) ≠ 1), and **t** will point in a wrong direction.

---

### Step 5 — Decompose E into Four Candidate (R, t) Pairs

Using the corrected **E = U · diag(1,1,0) · Vᵀ**, define:

```
W = | 0  -1   0 |
    | 1   0   0 |
    | 0   0   1 |
```

The four mathematically valid solutions are:

| Solution | R | t |
|:---:|:---:|:---:|
| 1 | U W Vᵀ | +u₃ |
| 2 | U W Vᵀ | −u₃ |
| 3 | U Wᵀ Vᵀ | +u₃ |
| 4 | U Wᵀ Vᵀ | −u₃ |

where **u₃** is the third column of **U** (corresponding to the zero singular value).

All four satisfy the epipolar constraint mathematically. The next step selects the physically meaningful one.

---

### Step 6 — Cheirality Check

**Cheirality** means "the 3D point must be in front of both cameras." Among the four solutions, only one places reconstructed points in front of both cameras simultaneously.

The procedure:
1. Take one (or a few) known point correspondence.
2. Triangulate a 3D point **X** using each of the four (R, t) pairs.
3. Compute the depth of **X** in both camera frames.
4. Select the (R, t) for which **both depths are positive**.

```python
_, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)
# recoverPose handles Step 5 and Step 6 automatically
```

---

## 4. Putting It All Together — Code Example

```python
import cv2
import numpy as np

# --- Inputs ---
# img1, img2: your two images
# K: camera intrinsic matrix (3x3), assumed same for both cameras

# Step 1: Feature detection and matching
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test to filter weak matches
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# Step 2: Convert to normalized coordinates
def normalize_pts(pts, K):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])  # homogeneous
    return (np.linalg.inv(K) @ pts_h.T).T[:, :2]

pts1_n = normalize_pts(pts1, K)
pts2_n = normalize_pts(pts2, K)

# Step 3: Estimate E with RANSAC
E, mask = cv2.findEssentialMat(
    pts1_n, pts2_n,
    focal=1.0, pp=(0., 0.),
    method=cv2.RANSAC, prob=0.999, threshold=1e-3
)

# Step 4: Enforce rank-2 (done internally by findEssentialMat,
#          but shown here for clarity)
U, S, Vt = np.linalg.svd(E)
E = U @ np.diag([1, 1, 0]) @ Vt

# Steps 5 & 6: Decompose E and select correct (R, t)
_, R, t, _ = cv2.recoverPose(E, pts1_n, pts2_n)

print("Rotation R:\n", R)
print("Translation direction t:\n", t)
```

---

## 5. Common Pitfalls

| Pitfall | What goes wrong | Fix |
|---|---|---|
| Skipping RANSAC | **E** is completely wrong due to outlier matches | Always use `method=cv2.RANSAC` |
| Wrong input to findEssentialMat | Passing pixel coordinates instead of normalized | Apply K⁻¹ first, or pass K directly to the function |
| Skipping rank-2 correction | **R** is not a valid rotation matrix | Apply SVD correction before decomposition |
| Ignoring cheirality | Picking the wrong (R, t) from the four candidates | Use `recoverPose()` which handles this automatically |
| Assuming t has scale | Applying the recovered t as a metric distance | Remember: t is a unit vector — scale is unknown |

---

## 6. Where This Is Used in Practice

| Application | How (R, t) is used |
|---|---|
| **Structure from Motion (SfM)** e.g., COLMAP | Chain (R, t) across many image pairs to reconstruct full camera trajectories and sparse 3D point clouds |
| **Visual Odometry / SLAM** e.g., ORB-SLAM | Estimate incremental camera motion frame-by-frame for robot navigation |
| **NeRF / 3D Gaussian Splatting** | These methods take pre-computed (R, t) from COLMAP as input — they do not estimate pose themselves |
| **Augmented Reality** | Register virtual objects to the real world by knowing the camera pose |

---

*This is supplementary material not covered in the lecture slides. For the mathematical derivation of the four (R, t) solutions, refer to: Hartley & Zisserman, "Multiple View Geometry in Computer Vision," Chapter 9.*
