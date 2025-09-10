import cameragetter
from blobwatch import Blobwatch, blobwatch_process, blobwatch_release_observation
import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
from typing import List, Dict, Optional
from  buteforcesolver import SolvePoseBruteOptimized

class BlobwatchLEDTracker:    
    def __init__(self):
        #camera setup
        self.camera = cameragetter.D455Camera()
        self.camera.set_exposure(50)
        
        #camera intrinsics
        rs_intrinsics = self.camera.get_camera_intrinsics()
        self.camera_matrix = np.array([
            [rs_intrinsics.fx, 0, rs_intrinsics.ppx],
            [0, rs_intrinsics.fy, rs_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        #distortion coefficients from RealSense
        self.dist_coeffs = np.array([
            rs_intrinsics.coeffs[0],  #k1
            rs_intrinsics.coeffs[1],  #k2
            rs_intrinsics.coeffs[2],  #p1
            rs_intrinsics.coeffs[3],  #p2
            rs_intrinsics.coeffs[4]   #k3
        ], dtype=np.float64).reshape(-1, 1)
        
        print(f"Camera Matrix:\n{self.camera_matrix}")
        print(f"Distortion Coeffs: {self.dist_coeffs.flatten()}")
        
        #blob detector
        self.blobwatch = Blobwatch(pixel_threshold=100, blob_required_threshold=200)
        
        #LED world points
        self.world_points = np.array([
            (10, -10, 0), (30, -10, 0), (70, -10, 0),
            (20, -20, 0),
            (10, -30, 0), (50, -30, 0), (70, -30, 0),
            (20, -40, 0), (60, -40, 0),
            (30, -50, 0),
            (20, -60, 0),(60, -60, 0),
        ], dtype=np.float64)
        
        # Pose solver - using multiprocess version
        self.pose_solver = SolvePoseBruteOptimized(
            self.camera_matrix, 
            self.dist_coeffs, 
            self.world_points,
            adaptive_search=True,
            distance_adaptive=True,
            use_temporal=True,
            max_iterations=5000,
            stop_on_strong = False,
        )

        self.blob_tracker = SimpleBlobTracker(max_distance=50.0)
        
        self.detection_queue = queue.Queue(maxsize=3)
        self.pose_queue = queue.Queue(maxsize=2)
        self.display_queue = queue.Queue(maxsize=2)
        
        self.state_lock = threading.Lock()
        self.latest_frame = None
        self.latest_detections = []
        self.latest_pose_result = None
        self.running = False
        self.threads = []
        
        self.stats = {
            'frame_fps': 0.0,
            'detection_fps': 0.0,
            'pose_fps': 0.0,
            'display_fps': 0.0,
            'total_detections': 0,
            'successful_poses': 0,
            'failed_poses': 0
        }
        self.fps_counters = {
            'frame': deque(maxlen=30),
            'detection': deque(maxlen=30),
            'pose': deque(maxlen=10),
            'display': deque(maxlen=30)
        }

    def start(self):
        self.running = True
        
        self.threads = [
            threading.Thread(target=self._camera_detection_thread, name="CameraDetection"),
            threading.Thread(target=self._pose_solving_thread, name="PoseSolver"),
            threading.Thread(target=self._display_thread, name="Display"),
        ]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
            print(f"Started {thread.name} thread")

    def stop(self):
        print("Stopping threads...")
        self.running = False
        
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        cv2.destroyAllWindows()
        print("All threads stopped")

    def _camera_detection_thread(self):
        print("Camera and Detection thread started")
        frame_count = 0
        
        while self.running:
            try:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                observation = blobwatch_process(self.blobwatch, frame)
                
                #convert Blobwatch blobs to detection format
                detections = []
                if observation is not None:
                    for i in range(observation.num_blobs):
                        blob = observation.blobs[i]
                        
                        detection = {
                            'x': float(blob['x']),
                            'y': float(blob['y']),
                            'inner': float(min(blob['width'], blob['height'])), #FIXME this was from the HSF realated stuff and needs to be adapted
                            'outer': float(max(blob['width'], blob['height'])),
                            'left': int(blob['left']),
                            'top': int(blob['top']),
                            'width': int(blob['width']),
                            'height': int(blob['height']),
                            'blob_id': int(blob['blob_id']),
                            'area': float(blob['width'] * blob['height'])
                        }
                        detections.append(detection)
                    
                    blobwatch_release_observation(self.blobwatch, observation)
                
                #update blob tracker for ID persistence
                tracked_detections = self.blob_tracker.update(detections)
                
                current_time = time.time()
                self.fps_counters['frame'].append(current_time)
                self.fps_counters['detection'].append(current_time)
                
                with self.state_lock:
                    self.latest_frame = frame.copy()
                    self.latest_detections = tracked_detections.copy()
                    self.stats['total_detections'] = len(tracked_detections)
                
                frame_count += 1
                
                if len(tracked_detections) >= 4:
                    try:
                        self.detection_queue.put_nowait({
                            'detections': tracked_detections.copy(),
                            'frame_shape': frame.shape,
                            'frame_count': frame_count
                        })
                    except queue.Full:
                        pass
                
                try:
                    self.display_queue.put_nowait({
                        'frame': frame.copy(),
                        'detections': tracked_detections.copy(),
                        'frame_count': frame_count
                    })
                except queue.Full:
                    pass
                
            except Exception as e:
                print(f"Camera+Detection thread error: {e}")
                time.sleep(0.01)

    def _pose_solving_thread(self):
        """Pose solving thread using multiprocess solver"""
        print("PoseSolver thread started")
        
        while self.running:
            try:
                data = self.detection_queue.get(timeout=0.1)
                detections = data['detections']
                frame_shape = data['frame_shape']
                frame_count = data['frame_count']
                
                start_time = time.time()
                ok, rvec, tvec, score = self.pose_solver.solve(
                    detections, frame_shape, assume_top_left=False  #blobwatch gives centroids
                )
                solve_time = time.time() - start_time
                
                self.fps_counters['pose'].append(time.time())
                
                with self.state_lock:
                    if ok:
                        self.stats['successful_poses'] += 1
                    else:
                        self.stats['failed_poses'] += 1
                
                result = {
                    'ok': ok,
                    'rvec': rvec,
                    'tvec': tvec,
                    'score': score,
                    'solve_time': solve_time,
                    'frame_count': frame_count,
                    'detections_used': len(detections)
                }
                
                try:
                    self.pose_queue.put_nowait(result)
                except queue.Full:
                    pass
                
                print(f"Pose solve #{self.stats['successful_poses'] + self.stats['failed_poses']}: "
                      f"{'SUCCESS' if ok else 'FAILED'} in {solve_time:.3f}s, "
                      f"inliers={score.get('inliers', 0)}, "
                      f"rmse={score.get('reproj_rmse', 0):.2f}px")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"PoseSolver thread error: {e}")
                time.sleep(0.01)

    def _display_thread(self):
        """Display thread with visualization"""
        print("Display thread started")
        latest_pose_result = None
        
        while self.running:
            try:
                display_data = self.display_queue.get(timeout=0.1)
                frame = display_data['frame']
                detections = display_data['detections']
                frame_count = display_data['frame_count']
                
                try:
                    while True:
                        latest_pose_result = self.pose_queue.get_nowait()
                        with self.state_lock:
                            self.latest_pose_result = latest_pose_result
                except queue.Empty:
                    pass
                
                self.fps_counters['display'].append(time.time())
                
                with self.state_lock:
                    stats = self.stats.copy()
                    
                for name, counter in self.fps_counters.items():
                    if len(counter) > 1:
                        time_span = counter[-1] - counter[0]
                        stats[f'{name}_fps'] = (len(counter) - 1) / max(time_span, 0.001)
                
                out = self._draw_detections_and_pose(frame, detections, latest_pose_result)
                
                self._draw_performance_overlay(out, stats, frame_count)
                
                cv2.imshow("Blobwatch LED Tracking", out)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('s'):
                    filename = f"led_frame_{int(time.time())}.png"
                    cv2.imwrite(filename, out)
                    print(f"Saved frame to {filename}")
                elif key == ord('h'):
                    print("\nControls:")
                    print("  q - quit")
                    print("  s - save frame")
                    print("  h - show this help")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Display thread error: {e}")
                time.sleep(0.01)

    def _draw_detections_and_pose(self, frame, detections, pose_result):
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        led_correspondences = {}
        if pose_result and pose_result['ok']:
            score = pose_result['score']
            led_correspondences = score.get('led_correspondences', {})
        
        detection_to_led = {}
        if led_correspondences:
            for led_idx, det_idx in led_correspondences.items():
                detection_to_led[det_idx] = led_idx
        
        for i, detection in enumerate(detections):
            left = detection['left']
            top = detection['top']
            width = detection['width']
            height = detection['height']
            cx, cy = int(detection['x']), int(detection['y'])
            track_id = detection.get('id', detection.get('blob_id', i))
            
            if i in detection_to_led:
                bbox_color = (0, 255, 0)  #green for matched LEDs
                center_color = (0, 255, 255)
                text_color = (0, 255, 255)
            else:
                bbox_color = (128, 128, 128)  #gray for unmatched
                center_color = (0, 0, 255)  
                text_color = (128, 128, 128)
            
            cv2.rectangle(out, (left, top), (left + width, top + height), bbox_color, 1)
            
            cv2.circle(out, (cx, cy), 3, center_color, -1)
            
            cv2.drawMarker(out, (cx, cy), center_color, markerType=cv2.MARKER_CROSS, 
                          markerSize=8, thickness=1, line_type=cv2.LINE_AA)
            
            #label 
            if i in detection_to_led:
                led_idx = detection_to_led[i]
                label = f"ID{track_id} (LED{led_idx})"
            else:
                label = f"ID{track_id}"
            
            cv2.putText(out, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, text_color, 1, cv2.LINE_AA)
        
        #draw pose
        if pose_result and pose_result['ok']:
            rvec, tvec = pose_result['rvec'], pose_result['tvec']
            
            try:
                cv2.drawFrameAxes(out, self.camera_matrix, self.dist_coeffs, rvec, tvec, 20)
            except Exception as e:
                print(f"Error drawing axes: {e}")
            
            try:
                proj_points, _ = cv2.projectPoints(self.world_points, rvec, tvec, 
                                                 self.camera_matrix, self.dist_coeffs)
                proj_points = proj_points.reshape(-1, 2)
                
                for j, point in enumerate(proj_points):
                    px, py = int(point[0]), int(point[1])
                    if 0 <= px < out.shape[1] and 0 <= py < out.shape[0]:
                        cv2.circle(out, (px, py), 4, (255, 0, 255), 1)  # Magenta circles
                        cv2.putText(out, f"L{j}", (px + 5, py - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
            except Exception as e:
                print(f"Error projecting world points: {e}")
        
        return out

    def _draw_performance_overlay(self, img, stats, frame_count):
        h, w = img.shape[:2]
        
        if self.latest_pose_result and self.latest_pose_result['ok']:
            score = self.latest_pose_result['score']
            cv2.putText(img, f"POSE: SUCCESS | inliers={score.get('inliers', 0)} | rmse={score.get('reproj_rmse', 0):.2f}px",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(img, f"solve_time={self.latest_pose_result.get('solve_time', 0):.3f}s | iterations={score.get('iterations_tested', 0)}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "POSE: NO SOLUTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.putText(img, f"FPS: cam={stats.get('frame_fps', 0):.1f} det={stats.get('detection_fps', 0):.1f} pose={stats.get('pose_fps', 0):.1f} disp={stats.get('display_fps', 0):.1f}",
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(img, f"DETECTIONS: {stats.get('total_detections', 0)} | POSES: success={stats.get('successful_poses', 0)} fail={stats.get('failed_poses', 0)}",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(img, f"FRAME: {frame_count}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


class SimpleBlobTracker:
    
    def __init__(self, max_distance=50.0):
        self.max_distance = max_distance
        self.tracks = {}
        self.next_id = 1
    
    def update(self, detections):
        """Update tracks and assign persistent IDs"""
        current_positions = [(d['x'], d['y']) for d in detections]
        
        used_tracks = set()
        for i, detection in enumerate(detections):
            pos = (detection['x'], detection['y'])
            
            # Find closest track
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, last_pos in self.tracks.items():
                if track_id in used_tracks:
                    continue
                    
                distance = np.sqrt((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            #assign track ID
            if best_track_id is not None:
                detection['id'] = best_track_id
                used_tracks.add(best_track_id)
                self.tracks[best_track_id] = pos
            else:
                #new track
                detection['id'] = self.next_id
                self.tracks[self.next_id] = pos
                self.next_id += 1
        
        active_tracks = {d['id'] for d in detections}
        dead_tracks = set(self.tracks.keys()) - active_tracks
        for track_id in dead_tracks:
            del self.tracks[track_id]
        
        return detections


def main():
    tracker = BlobwatchLEDTracker()
    
    try:
        tracker.start()
        print("Blobwatch LED tracker started!")
        print("Press 'h' for help, 'q' to quit")
        
        while tracker.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()  