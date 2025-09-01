import numpy as np
import cv2 as cv

from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time as RosTime


def _make_object_points(board_size, square_m):
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # row-major
    objp[:, :2] = grid * square_m
    return objp


def detect_checkerboard_pose_T_GC(
        img_bgr, K, dist, board_size, square_m
):
    """
    Returns 4x4 T_GC (camera pose in Grid frame) if found; raises RuntimeError otherwise.
    Conventions:
      - Grid frame G: z-axis is board normal. We set R_WG = I and G at world origin.
      - OpenCV returns rvec,tvec for T_CG (grid->camera). We invert to get T_GC.
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    flags = (cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE)
    found, corners = cv.findChessboardCorners(gray, board_size, flags)
    if not found:
        raise RuntimeError("Checkerboard not found")

    # Subpixel refine
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    objp = _make_object_points(board_size, square_m)

    ok, rvec, tvec, inliers = cv.solvePnPRansac(
        objp, corners, K, dist,
        flags=cv.SOLVEPNP_ITERATIVE,
        iterationsCount=100, reprojectionError=2.0, confidence=0.999
    )
    if not ok:
        raise RuntimeError("solvePnPRansac failed")

    # Optional refinement if available
    if inliers is not None and hasattr(cv, "solvePnPRefineLM"):
        idx = inliers[:,0]
        rvec, tvec = cv.solvePnPRefineLM(objp[idx], corners[idx], K, dist, rvec, tvec)

    R_CG, _ = cv.Rodrigues(rvec)      # grid->camera rotation
    t_CG = tvec.reshape(3,1)          # grid->camera translation

    # Invert to camera->grid
    R_GC = R_CG.T
    t_GC = -R_CG.T @ t_CG

    T_GC = np.eye(4)
    T_GC[:3,:3] = R_GC
    T_GC[:3, 3] = t_GC[:,0]
    return T_GC


def T_to_transform_stamped(T_WC, stamp: RosTime, world_frame: str, camera_frame: str) -> TransformStamped:
    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = world_frame
    ts.child_frame_id = camera_frame

    # translation
    ts.transform.translation.x = float(T_WC[0,3])
    ts.transform.translation.y = float(T_WC[1,3])
    ts.transform.translation.z = float(T_WC[2,3])

    # rotation (matrix -> quaternion)
    R = T_WC[:3,:3]
    qw = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    qx = (R[2,1] - R[1,2]) / (4.0*qw) if qw != 0 else 0.0
    qy = (R[0,2] - R[2,0]) / (4.0*qw) if qw != 0 else 0.0
    qz = (R[1,0] - R[0,1]) / (4.0*qw) if qw != 0 else 0.0
    ts.transform.rotation.x = float(qx)
    ts.transform.rotation.y = float(qy)
    ts.transform.rotation.z = float(qz)
    ts.transform.rotation.w = float(qw)
    return ts
