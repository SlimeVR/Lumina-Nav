import cv2 as cv
import glob
import numpy as np
import os
from scipy import linalg

camera_device_id = "cameras.avi"
frame_width = 2560
frame_height = 720

mono_calibration_frames = 10
stereo_calibration_frames = 10

view_resize = 2
cooldown = 2
checkerboard_rows = 10
checkerboard_columns = 7
checkerboard_box_size_scale = 21.5

def DLT(P1, P2, point1, point2):
    A = [
        point1[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point1[0]*P1[2,:],
        point2[1]*P2[2,:] - P2[1,:],
        P2[0,:] - point2[0]*P2[2,:]
    ]
    A = np.array(A).reshape((4,4))
    B = A.T @ A
    _, _, Vh = linalg.svd(B, full_matrices=False)
    X = Vh[3,0:3]/Vh[3,3]
    return X

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1.0
    return P

def get_projection_matrix(cmtx, R, T):
    return cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]

def save_stacked_frames(mode="mono"):
    if mode == "mono":
        out_dir = "frames"
        num_to_save = mono_calibration_frames
    else:
        out_dir = "frames_pair"
        num_to_save = stereo_calibration_frames
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    cap = cv.VideoCapture(camera_device_id)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    cooldown_counter = cooldown * 30
    start = False
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No video data received. Exiting capture...")
            break
        
        left_frame = frame[:, :1280]
        right_frame = frame[:, 1280:]

        combined_small = cv.hconcat([
            cv.resize(left_frame, None, fx=1/view_resize, fy=1/view_resize),
            cv.resize(right_frame, None, fx=1/view_resize, fy=1/view_resize)
        ])
        
        if not start:
            cv.putText(combined_small,
                       "Press SPACE to start saving frames",
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        else:
            cooldown_counter -= 1
            cv.putText(combined_small, f"Cooldown: {cooldown_counter}",
                       (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(combined_small, f"Num frames: {saved_count}",
                       (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

            if cooldown_counter <= 0:
                left_name = os.path.join(out_dir, f"camera0_{saved_count}.png")
                right_name = os.path.join(out_dir, f"camera1_{saved_count}.png")
                cv.imwrite(left_name, left_frame)
                cv.imwrite(right_name, right_frame)
                saved_count += 1
                cooldown_counter = cooldown * 30
        
        cv.imshow(f"{mode}_calibration", combined_small)
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            start = True
        
        if saved_count >= num_to_save:
            break
    
    cap.release()
    cv.destroyAllWindows()

def calibrate_camera_for_intrinsic_parameters(images_prefix):
    images_names = glob.glob(images_prefix)
    images = [cv.imread(name, 1) for name in images_names if cv.imread(name, 1) is not None]

    if len(images) == 0:
        print(f"No images found for prefix: {images_prefix}")
        return None, None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = checkerboard_rows
    cols = checkerboard_columns
    world_scaling = checkerboard_box_size_scale

    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp *= world_scaling

    objpoints = []
    imgpoints = []

    h, w = images[0].shape[:2]

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, cols), None)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            tmp_show = img.copy()
            cv.drawChessboardCorners(tmp_show, (rows, cols), corners, ret)
            cv.putText(tmp_show, 'Press "s" to skip if corners are bad',
                       (30, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv.imshow("Mono calibration check", tmp_show)
            k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
                print("Skipping this frame...")
                continue
            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()

    if len(objpoints) == 0:
        print("No checkerboard detections. Returning None.")
        return None, None

    ret, cmtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    print(f"Camera calibration RMSE: {ret}")
    print(f"Camera matrix:\n{cmtx}")
    print(f"Distortion:\n{dist}")
    return cmtx, dist

def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')
    filename = os.path.join('camera_parameters', f"{camera_name}_intrinsics.dat")
    with open(filename, 'w') as f:
        f.write("intrinsic:\n")
        for row in camera_matrix:
            f.write(" ".join(str(v) for v in row) + "\n")
        f.write("distortion:\n")
        f.write(" ".join(str(v) for v in distortion_coefs[0]) + "\n")
    print(f"Saved {camera_name} intrinsics to {filename}")

def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    c0_names = sorted(glob.glob(frames_prefix_c0))
    c1_names = sorted(glob.glob(frames_prefix_c1))

    c0_images = [cv.imread(n, 1) for n in c0_names]
    c1_images = [cv.imread(n, 1) for n in c1_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows = checkerboard_rows
    cols = checkerboard_columns
    world_scaling = checkerboard_box_size_scale

    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp *= world_scaling

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    if len(c0_images) == 0:
        print("No pairs found for stereo calibration!")
        return np.eye(3), np.zeros((3,1))
    h, w = c0_images[0].shape[:2]

    for imgL, imgR in zip(c0_images, c1_images):
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        retL, cornersL = cv.findChessboardCorners(grayL, (rows, cols), None)
        retR, cornersR = cv.findChessboardCorners(grayR, (rows, cols), None)

        if retL and retR:
            cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            
            tmpL = imgL.copy()
            tmpR = imgR.copy()
            cv.drawChessboardCorners(tmpL, (rows,cols), cornersL, retL)
            cv.drawChessboardCorners(tmpR, (rows,cols), cornersR, retR)
            cv.imshow("Stereo Left", tmpL)
            cv.imshow("Stereo Right", tmpR)
            k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
                print("Skipping this stereo pair...")
                continue

            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)
    cv.destroyAllWindows()

    if len(objpoints) == 0:
        print("No valid pairs for stereo calibrations. Returning identity.")
        return np.eye(3), np.zeros((3,1))

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx0, dist0, mtx1, dist1,
        (w, h),
        criteria=criteria,
        flags=stereocalibration_flags
    )

    print(f"Stereo calibration RMSE: {ret}")
    print("R:\n", R)
    print("T:\n", T)
    return R, T

def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=''):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')
    
    out0 = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    with open(out0, 'w') as f0:
        f0.write("R:\n")
        for row in R0:
            f0.write(" ".join(str(v) for v in row) + "\n")
        f0.write("T:\n")
        for val in T0:
            f0.write(" ".join(str(v) for v in val) + "\n")

    out1 = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    with open(out1, 'w') as f1:
        f1.write("R:\n")
        for row in R1:
            f1.write(" ".join(str(v) for v in row) + "\n")
        f1.write("T:\n")
        for val in T1:
            f1.write(" ".join(str(v) for v in val) + "\n")

    print(f"Saved extrinsics:\n  {out0}\n  {out1}")

def check_calibration(cam0_data, cam1_data, _zshift=50.):
    cmtx0, dist0, R0, T0 = cam0_data
    cmtx1, dist1, R1, T1 = cam1_data

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    axes_3d = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32)
    axes_3d *= 5
    axes_3d[:, 2] += _zshift

    cap = cv.VideoCapture(camera_device_id)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame in check_calibration()!")
            break

        left = frame[:, :1280]
        right = frame[:, 1280:]

        pts0 = []
        for pt in axes_3d:
            uvw = P0 @ np.array([pt[0], pt[1], pt[2], 1.])
            pts0.append( (uvw[0]/uvw[2], uvw[1]/uvw[2]) )
        
        pts1 = []
        for pt in axes_3d:
            uvw = P1 @ np.array([pt[0], pt[1], pt[2], 1.])
            pts1.append( (uvw[0]/uvw[2], uvw[1]/uvw[2]) )
        
        origin0 = (int(pts0[0][0]), int(pts0[0][1]))
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        for col, pt_ in zip(colors, pts0[1:]):
            endp = (int(pt_[0]), int(pt_[1]))
            cv.line(left, origin0, endp, col, 2)

        origin1 = (int(pts1[0][0]), int(pts1[0][1]))
        for col, pt_ in zip(colors, pts1[1:]):
            endp = (int(pt_[0]), int(pt_[1]))
            cv.line(right, origin1, endp, col, 2)

        combined = cv.hconcat([
            cv.resize(left, None, fx=1/view_resize, fy=1/view_resize),
            cv.resize(right, None, fx=1/view_resize, fy=1/view_resize)
        ])

        cv.imshow("Extrinsic Check (Press ESC to quit)", combined)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":

    print("=== Step 1: Capture frames for single-camera (mono) calibration ===")
    save_stacked_frames(mode="mono")

    print("\n=== Step 2: Calibrate camera0 and camera1 intrinsics ===")
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters("frames/camera0_*")
    save_camera_intrinsics(cmtx0, dist0, "camera0")

    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters("frames/camera1_*")
    save_camera_intrinsics(cmtx1, dist1, "camera1")

    print("\n=== Step 3: Capture frames for stereo calibration ===")
    save_stacked_frames(mode="stereo")

    print("\n=== Step 4: Stereo calibration ===")
    R_01, T_01 = stereo_calibrate(cmtx0, dist0, cmtx1, dist1,
                                  "frames_pair/camera0_*",
                                  "frames_pair/camera1_*")

    R0 = np.eye(3, dtype=np.float32)
    T0 = np.zeros((3, 1), dtype=np.float32)

    print("\n=== Saving extrinsic parameters ===")
    save_extrinsic_calibration_parameters(R0, T0, R_01, T_01)

    print("\n=== Visual check of calibration ===")
    cam0_data = [cmtx0, dist0, R0, T0]
    cam1_data = [cmtx1, dist1, R_01, T_01]
    check_calibration(cam0_data, cam1_data, _zshift=50.)

    print("\nAll done!")
