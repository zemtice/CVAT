"""
Image Stitching via Automatic Keypoints Matching
================================================
Automatically stitches two images captured from different viewpoints
of the same planar scene using SIFT + FLANN + RANSAC + Homography.

Usage:
    python stitch_images.py --left left.jpg --right right.jpg
    python stitch_images.py --left left.jpg --right right.jpg --output result.jpg
    python stitch_images.py --demo          # run with synthetic demo images

Pipeline:
    1. Load two input images
    2. Detect keypoints and compute SIFT descriptors
    3. Match descriptors with FLANN k-NN matcher
    4. Filter matches using Lowe's ratio test
    5. Estimate Homography H (right -> left coordinates) with RANSAC
    6. Warp the right image into the left image's coordinate frame
    7. Blend and save the stitched panorama
"""

import argparse
import sys
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────
# 1. Feature Detection and Descriptor Computation
# ─────────────────────────────────────────────────────────────
def detect_and_compute(img_gray: np.ndarray):
    """
    Detect keypoints and compute SIFT descriptors on a grayscale image.

    Args:
        img_gray: Grayscale input image of shape (H, W).

    Returns:
        kps   : List of cv2.KeyPoint objects.
        descs : Descriptor array of shape (N, 128), dtype float32.
    """
    sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(img_gray, None)
    return kps, descs


# ─────────────────────────────────────────────────────────────
# 2. Feature Matching (FLANN + Lowe's Ratio Test)
# ─────────────────────────────────────────────────────────────
def match_features(descs1: np.ndarray, descs2: np.ndarray,
                   ratio: float = 0.75):
    """
    Match descriptors using FLANN KD-tree k-NN (k=2) and filter with
    Lowe's ratio test.

    Lowe's ratio test:
        Keep a match only if  d(best) / d(second_best) < ratio.
        This rejects ambiguous matches whose nearest neighbor in
        descriptor space is not clearly closer than the second nearest.

    Args:
        descs1 : Descriptors from image 1.
        descs2 : Descriptors from image 2.
        ratio  : Lowe's ratio threshold (default 0.75).

    Returns:
        good : List of cv2.DMatch objects that passed the ratio test.
    """
    # algorithm=1 selects FLANN_INDEX_KDTREE; trees=5 balances speed/accuracy
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descs1, descs2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


# ─────────────────────────────────────────────────────────────
# 3. Robust Homography Estimation via RANSAC
# ─────────────────────────────────────────────────────────────
def compute_homography(kps1, kps2, good_matches,
                       reproj_thresh: float = 4.0):
    """
    Robustly estimate a Homography H from matched point pairs using RANSAC,
    such that:   x1 ≈ H · x2   (maps image-2 points into image-1 coordinates).

    A Homography is a 3x3 projective transformation matrix. It fully
    describes the mapping between two views of the same planar surface,
    or between views related by a pure camera rotation (no translation).

    Args:
        kps1          : Keypoints from image 1.
        kps2          : Keypoints from image 2.
        good_matches  : Filtered match list from match_features().
        reproj_thresh : RANSAC reprojection error threshold in pixels
                        (a point is an inlier if its reprojection error
                        is below this value).

    Returns:
        H    : 3x3 Homography matrix (np.ndarray), or None on failure.
        mask : Inlier/outlier mask of shape (N, 1), dtype uint8.
    """
    if len(good_matches) < 4:
        print(f"[ERROR] Not enough good matches ({len(good_matches)} < 4) "
              f"to compute Homography.")
        return None, None

    # Extract matched point coordinates
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches])

    # findHomography expects (src, dst); here src=pts2, dst=pts1
    H, mask = cv2.findHomography(pts2, pts1,
                                 cv2.RANSAC, reproj_thresh)
    return H, mask


# ─────────────────────────────────────────────────────────────
# 4. Perspective Warp and Blending
# ─────────────────────────────────────────────────────────────
def warp_and_blend(img1: np.ndarray, img2: np.ndarray,
                   H: np.ndarray) -> np.ndarray:
    """
    Warp img2 into img1's coordinate frame using Homography H, then
    merge both images onto a common canvas.

    Steps:
      a) Project the four corners of img2 through H to find their
         positions in img1's coordinate frame; compute canvas bounds.
      b) If any corner lands at negative coordinates (img2 extends
         left of img1), prepend a translation T so all pixels are
         at non-negative coordinates.
      c) Apply warpPerspective to img2 with the shifted homography T @ H.
      d) Paste img1 onto the canvas at its offset position (tx, ty).
      e) In the overlap region, blend both images with equal 50/50 weights.

    Args:
        img1 : Reference (left) image, shape (H1, W1, 3).
        img2 : Query (right) image to be warped, shape (H2, W2, 3).
        H    : Homography mapping img2 coordinates -> img1 coordinates.

    Returns:
        panorama : Stitched output image.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Project the four corners of img2 into img1's coordinate frame
    corners2 = np.float32([[0, 0], [w2, 0],
                            [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners2, H)

    # Combine with img1's own corners to compute the full canvas extent
    corners1 = np.float32([[0, 0], [w1, 0],
                            [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners1, warped_corners], axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    # Translation offset to shift all coordinates into the positive quadrant
    tx = max(-x_min, 0)
    ty = max(-y_min, 0)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Build shifted homography: apply translation on top of H
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float64)
    H_shifted = T @ H
    warped2 = cv2.warpPerspective(img2, H_shifted, (canvas_w, canvas_h))

    # Place img1 onto the canvas at its offset position
    canvas = warped2.copy()
    canvas[ty:ty + h1, tx:tx + w1] = img1

    # Identify the overlap region between img1's footprint and warped img2
    mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask1[ty:ty + h1, tx:tx + w1] = 255

    mask2 = (warped2.sum(axis=2) > 0).astype(np.uint8) * 255
    overlap = cv2.bitwise_and(mask1, mask2)

    # Blend overlap with equal weights (simple average)
    if overlap.any():
        region1 = canvas[overlap > 0].astype(np.float32)
        region2 = warped2[overlap > 0].astype(np.float32)
        canvas[overlap > 0] = (0.5 * region1 + 0.5 * region2).astype(np.uint8)

    return canvas


# ─────────────────────────────────────────────────────────────
# 5. Match Visualization
# ─────────────────────────────────────────────────────────────
def draw_matches_result(img1, kps1, img2, kps2,
                        good_matches, mask=None) -> np.ndarray:
    """
    Visualize matched keypoints between two images.
    Inliers (accepted by RANSAC) are drawn in green;
    outliers are drawn in red when a mask is provided.

    Args:
        img1         : First image (BGR).
        kps1         : Keypoints of the first image.
        img2         : Second image (BGR).
        kps2         : Keypoints of the second image.
        good_matches : List of DMatch objects to draw.
        mask         : RANSAC inlier mask (N,1) uint8; None draws all green.

    Returns:
        vis : Side-by-side visualization image with match lines.
    """
    draw_params = dict(
        matchColor=(0, 255, 0),         # inliers -> green
        singlePointColor=(255, 0, 0),   # unmatched keypoints -> blue
        matchesMask=None,
        flags=cv2.DrawMatchesFlags_DEFAULT
    )

    if mask is not None:
        draw_params['matchesMask'] = mask.ravel().tolist()

    vis = cv2.drawMatches(img1, kps1, img2, kps2,
                          good_matches, None,
                          matchColor=draw_params['matchColor'],
                          singlePointColor=draw_params['singlePointColor'],
                          matchesMask=draw_params.get('matchesMask'),
                          flags=draw_params['flags'])
    return vis

# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Automatic Image Stitching via SIFT + RANSAC + Homography")
    parser.add_argument("--left",   type=str, help="Path to the left input image.")
    parser.add_argument("--right",  type=str, help="Path to the right input image.")
    parser.add_argument("--output", type=str, default="panorama.jpg",
                        help="Output panorama path (default: panorama.jpg).")
    parser.add_argument("--ratio",  type=float, default=0.75,
                        help="Lowe's ratio test threshold (default: 0.75).")
    parser.add_argument("--reproj", type=float, default=4.0,
                        help="RANSAC reprojection error threshold in pixels (default: 4.0).")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip GUI windows; only save output files.")
    args = parser.parse_args()

    # ── Load images ──
    if not args.left or not args.right:
        print("[ERROR] Please provide --left and --right image paths, or use --demo.")
        sys.exit(1)
    img1 = cv2.imread(args.left)
    img2 = cv2.imread(args.right)
    if img1 is None:
        print(f"[ERROR] Could not read image: {args.left}")
        sys.exit(1)
    if img2 is None:
        print(f"[ERROR] Could not read image: {args.right}")
        sys.exit(1)

    print(f"[INFO] Image 1 size: {img1.shape[1]}x{img1.shape[0]}")
    print(f"[INFO] Image 2 size: {img2.shape[1]}x{img2.shape[0]}")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ── Step 1: SIFT feature detection ──
    print("\n[Step 1] Detecting SIFT keypoints...")
    kps1, descs1 = detect_and_compute(gray1)
    kps2, descs2 = detect_and_compute(gray2)
    print(f"         Image 1: {len(kps1)} keypoints")
    print(f"         Image 2: {len(kps2)} keypoints")

    if descs1 is None or descs2 is None:
        print("[ERROR] Descriptor computation failed (zero keypoints detected).")
        sys.exit(1)

    # ── Step 2: FLANN matching + Lowe's ratio test ──
    print(f"\n[Step 2] FLANN matching (ratio={args.ratio})...")
    good = match_features(descs1, descs2, ratio=args.ratio)
    print(f"         Good matches after ratio test: {len(good)}")

    # ── Step 3: RANSAC Homography estimation ──
    print(f"\n[Step 3] Estimating Homography with RANSAC (reproj_thresh={args.reproj} px)...")
    H, mask = compute_homography(kps1, kps2, good, reproj_thresh=args.reproj)
    if H is None:
        sys.exit(1)

    n_inlier  = int(mask.sum()) if mask is not None else 0
    n_outlier = len(good) - n_inlier
    print(f"         Inliers: {n_inlier},  Outliers: {n_outlier}")
    print(f"\n         Homography H (img2 -> img1 coordinate frame):")
    print(np.array2string(H, precision=4, suppress_small=True, prefix="         "))

    # ── Step 4: Warp and blend ──
    print("\n[Step 4] Warping and blending images...")
    panorama = warp_and_blend(img1, img2, H)
    print(f"         Panorama size: {panorama.shape[1]}x{panorama.shape[0]}")

    # ── Step 5: Visualize correspondences ──
    vis_matches = draw_matches_result(img1, kps1, img2, kps2, good, mask)

    # ── Save results ──
    cv2.imwrite(args.output, panorama)
    cv2.imwrite("matches.jpg", vis_matches)
    print(f"\n[INFO] Panorama saved: {args.output}")
    print(f"[INFO] Match visualization saved: matches.jpg")

    # ── Display results ──
    if not args.no_display:
        cv2.imshow("Matches (Green=Inlier)", vis_matches)
        cv2.imshow("Panorama", panorama)
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n[DONE] Stitching complete!")


if __name__ == "__main__":
    main()
