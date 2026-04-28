import numpy as np
import cv2

def get_normalization_matrix(pts):
    ## Normalizing transformations (T)
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    scale = np.sqrt(2) / std
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    return T

def solve_dlt(src_pts, dst_pts):
    # Normalize points for stability
    T1 = get_normalization_matrix(src_pts)
    T2 = get_normalization_matrix(dst_pts)
    
    # Apply normalization: x_norm = T * x_homogeneous
    src_h = np.column_stack([src_pts, np.ones(len(src_pts))])
    dst_h = np.column_stack([dst_pts, np.ones(len(dst_pts))])
    src_n = (T1 @ src_h.T).T
    dst_n = (T2 @ dst_h.T).T

    # Assemble Matrix A (2n x 9)
    A = []
    for i in range(len(src_pts)):
        x, y, _ = src_n[i]
        u, v, _ = dst_n[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A)

    # SVD: solution h is the last column of V
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :]
    H_norm = L.reshape(3, 3)

    # Denormalize: H = inv(T2) * H_norm * T1 (This is a GeminiAI clutch right here)
    H = np.linalg.inv(T2) @ H_norm @ T1
    return H / H[2, 2]

def unwrap_poster_no_target(image_path):
    img = cv2.imread(image_path)
    if img is None: return print("File not found.")
    h_img, w_img = img.shape[:2]

    # Define virtual correspondences using gmetry
    # 
    # pick 4 points on the cylinder (theta, height) and project them to camera (u, v)
    # target points (Ideal Unrolled Poster space)
    # map a 90-degree slice of the cylinder (-45 to +45 degrees)
    target_pts = np.array([
        [-np.pi/4,  1], # top left
        [ np.pi/4,  1], # top right
        [ np.pi/4, -1], # bot right
        [-np.pi/4, -1]  # bot left
    ])

    # Project these to Camera Image Plane (u, v)
    # Camera at (7,0,0), f=2. Image plane 1x1 (coords -0.5 to 0.5)
    camera_pts = []
    for theta, height in target_pts:
        # World Coords
        xw, yw, zw = np.cos(theta), np.sin(theta), height
        z_cam = 7.0 - xw
        x_cam = -yw
        y_cam = zw
        # Image Plane Coords (normalized -0.5 to 0.5) (second GeminiAI clutch)
        # without these all that is left is a 3D object but no way to "see" it on a 2D screen
        u_p = 2.0 * (x_cam / z_cam)
        v_p = 2.0 * (y_cam / z_cam)
        # Pixel Coords
        px = (u_p + 0.5) * w_img
        py = (0.5 - v_p) * h_img
        camera_pts.append([px, py])
    
    camera_pts = np.array(camera_pts)

    # Solve homography
    # define where we want these points in our output image (800x400)
    out_w, out_h = 800, 400
    output_canvas_pts = np.array([
        [0, 0], [out_w, 0], [out_w, out_h], [0, out_h]
    ], dtype=np.float32)

    H = solve_dlt(camera_pts, output_canvas_pts)

    # apply transformation
    # Note: As mentioned in the thread, because the surface is curved, a simple homography is an approximation.
    result = cv2.warpPerspective(img, H, (out_w, out_h))
    
    cv2.imwrite('unwrapped_result.jpg', result)
    print("Unwrapped image saved.")

unwrap_poster_no_target('camera view.png')
