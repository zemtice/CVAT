import numpy as np
import cv2

# Scene configuration
radius = 1
camera_location = (7, 0, 0)
z_range = (-1, +1)

# Calculate the visible range of pillar
angle = np.acos(radius/camera_location[0])
angle_range = (-angle, angle)

# Ratio and size of unwrap poster
w = angle*2
h = z_range[1] - z_range[0] 
pixel_per_unit = 50
w_px, h_px = int(w*pixel_per_unit), int(h*pixel_per_unit)
unwrap = np.zeros([h_px, w_px, 3], dtype=np.uint8)
print('Unwrap image size:', w_px, h_px)

# Camera extrinsic parameter (RT)
r_inv = np.array([[0, 0, -1],
                  [1, 0,  0], 
                  [0,-1,  0],], dtype=np.float32)
r = np.linalg.inv(r_inv)
t = np.array([[0], [0], [7]], dtype=np.float32)
rt = np.hstack([r, t])
print('Camera extrinsic RT =\n', rt)

# Camera intrinsic parameter (K)
k = np.array([[2000, 0, 500],
              [0, 2000, 500],
              [0, 0, 1]], dtype=np.float32)
print('Camera intrinsic K =\n', k)

# Precalculate the projection matrix P
krt = k@rt

# Load camera view image
cam_view = cv2.imread('camera view.png')
cam_view_draw = cam_view.copy() # Copy for visualization

# Project 3D points on pillar to imaging plane
for u in range(w_px):
    theta = angle_range[0]+u*(angle_range[1]-angle_range[0])/(w_px-1)
    x, y = np.cos(theta), np.sin(theta)
    for v in range(h_px):
        z = 1-v*(2)/(h_px-1)
        pos_3d = np.array([x, y, z, 1])
        pos_2d = krt@pos_3d
        pos_2d = pos_2d/pos_2d[-1]
        pos_2d = pos_2d[:2].astype(np.int32)

        # Assign color on camera view to the unwrap poster
        unwrap[v, u] = cam_view[pos_2d[1], pos_2d[0]]

        # Mark the sampling location on camera view
        cv2.circle(cam_view_draw, pos_2d, 1, (0, 0, 255), -1, cv2.LINE_4)
        cv2.imshow('sampling', cam_view_draw)

        cv2.imshow('unwrap', cv2.resize(unwrap, None, None, 4, 4, cv2.INTER_AREA))
        cv2.waitKey(1)

cv2.destroyAllWindows()

# Save result
cv2.imwrite('Discussion/D02/unwrap_image.png', unwrap)






        


