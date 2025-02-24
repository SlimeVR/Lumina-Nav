import cv2
import numpy as np
import matplotlib.pyplot as plt
from matchers import match_blobs_sumassign
from blobwatch import BlobWatch
from helperutils import load_intrinsics, load_extrinsics
from constutils import triangulate_dlt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], c='red', s=20)
plt.ion()
ax.set_xlim(-5, 5)
ax.set_ylim(-0, 5)
ax.set_zlim(-5, 5)
frame_width = 2560
frame_height = 720
split_width = 1280    



def main_demo():
    #init camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    #camera settings 
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


    #init blob find/track
    bwL = BlobWatch(pixel_threshold=100, min_area=5, max_area=500)
    bwR = BlobWatch(pixel_threshold=100, min_area=5, max_area=500)


    #load camera parameters from calibration
    K0, dist0 = load_intrinsics("camera_parameters/camera0_intrinsics.dat")
    K1, dist1 = load_intrinsics("camera_parameters/camera1_intrinsics.dat")

    R0, T0 = load_extrinsics("camera_parameters/camera0_rot_trans.dat")
    R1, T1 = load_extrinsics("camera_parameters/camera1_rot_trans.dat")

    # init projection matrices
    Rt0 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)  #the first camera has no rotation or translation (AKA the origin)
    P0 = K0 @ Rt0
    Rt1 = np.hstack([R1, T1])
    P1 = K1 @ Rt1

    #aLL the fun starts here :3
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #the stereo camera im sends both frames as a single frame (concatnated together) so here we split them 
        left_img = frame[:, :split_width]
        right_img = frame[:, split_width:] 

        grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        observationL = bwL.process_frame(grayL)
        observationR = bwR.process_frame(grayR)

        matches, unmatched_left, unmatched_right = match_blobs_sumassign(
            observationL.blobs,
            observationR.blobs,
            max_match_distance=500
        )
        combined = np.hstack((left_img.copy(), right_img.copy()))

        #just for visualization, no fun math happens here
        for blob in observationL.blobs:
            p1 = (int(blob.left), int(blob.top))
            p2 = (int(blob.left + blob.width), int(blob.top + blob.height))
            cv2.rectangle(combined, p1, p2, (0, 255, 0), 2)
            text_pos = (int(blob.x), int(blob.y))
            cv2.putText(
                combined,
                f"LID:{blob.blob_id}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )

        
        #also just for vis
        for blob in observationR.blobs:
            p1 = (int(blob.left + split_width), int(blob.top))
            p2 = (int(blob.left + blob.width + split_width), int(blob.top + blob.height))
            cv2.rectangle(combined, p1, p2, (255, 0, 0), 2)
            text_pos = (int(blob.x + split_width), int(blob.y))
            cv2.putText(
                combined,
                f"RID:{blob.blob_id}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1
            )

        #MORE VIS!! 
        for (i, j) in matches:
            left_blob = observationL.blobs[i]
            right_blob = observationR.blobs[j]
            pt_left = (int(left_blob.x), int(left_blob.y))
            pt_right = (int(right_blob.x + split_width), int(right_blob.y))
            cv2.line(combined, pt_left, pt_right, (0, 0, 255), 2)
    
        # I should prob turn this into its own function, but it converts camera space to camera space in meters
        dim3_points = []
        for (i, j) in matches:
            left_blob = observationL.blobs[i]
            right_blob = observationR.blobs[j]
            x0 = (left_blob.x, left_blob.y)
            x1 = (right_blob.x, right_blob.y)
            T1_meters = T1 / 1000.0
            Rt1 = np.hstack([R1, T1_meters])
            P1 = K1 @ Rt1
            X, Y, Z = triangulate_dlt(P0, P1, x0, x1)
            dim3_points.append((X, Y, Z))

        # This gets the points into the 3d visualazer
        dim3_points = np.array(dim3_points)
        if len(dim3_points) > 0:
            sc._offsets3d = (dim3_points[:, 0],
                             dim3_points[:, 2],
                             -dim3_points[:, 1])
            

        view_resize = 2 
        combined = cv2.resize(combined, (int(combined.shape[1] / view_resize), int(combined.shape[0] / view_resize)))
        cv2.imshow("Blob Matches", combined)
        plt.draw()
        plt.pause(0.01)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_demo()
