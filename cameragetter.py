import cv2 
import numpy as np
import pyrealsense2 as rs


class D455Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 60)    
        config.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 60)

        pipeline_profile = self.pipeline.start(config)

        device = pipeline_profile.get_device()
        depth_sensor = device.query_sensors()[0]
        set_emitter = 0
        depth_sensor.set_option(rs.option.emitter_enabled, set_emitter)
        emitter1 = depth_sensor.get_option(rs.option.emitter_enabled)


    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        ir_frame = frames.get_infrared_frame(1)
        ir_image = np.asanyarray(ir_frame.get_data())
        return ir_image
    def set_exposure(self, exposure_us):
        dev = self.pipeline.get_active_profile().get_device()
        stereo = None
        for s in dev.query_sensors():
            name = s.get_info(rs.camera_info.name)
            if name == "Stereo Module":
                stereo = s
                break
        if stereo is None:
            raise RuntimeError("Stereo Module not found")

        stereo.set_option(rs.option.enable_auto_exposure, 0)
        stereo.set_option(rs.option.exposure, float(exposure_us))
    
    def get_camera_intrinsics(self):
        
        profile = self.pipeline.get_active_profile()
        ir_stream = profile.get_stream(rs.stream.infrared, 1)
        intrinsics = ir_stream.as_video_stream_profile().get_intrinsics()
        return intrinsics
    
class cv2camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with index {camera_index}")
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray_frame
    def set_exposure(self, exposure_ms):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_ms))
    def get_camera_intrinsics(self):
        raise NotImplementedError("Camera intrinsics retrieval not implemented for cv2camera.")
    def release(self):
        self.cap.release()




if __name__ == "__main__":
    camera = D455Camera()
    camera.set_exposure(100)
    while True:
        frame = camera.get_frame()
        cv2.imshow("IR Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

