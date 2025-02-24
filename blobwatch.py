import cv2
import numpy as np
import logging
from collections import deque
from math import sqrt

logging.basicConfig(level=logging.DEBUG)

class Blob:
    def __init__(self, blob_id, x, y, area, top, left, width, height):
        self.blob_id = blob_id
        self.x = x
        self.y = y
        self.area = area
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.vx = 0.0
        self.vy = 0.0
        self.age = 0
        self.led_id = -1
        self.prev_led_id = -1
        self.track_index = -1

    def __repr__(self):
        return (f"Blob(ID={self.blob_id}, x={self.x:.2f}, y={self.y:.2f}, "
                f"area={self.area}, vx={self.vx:.2f}, vy={self.vy:.2f}, "
                f"age={self.age}, track_index={self.track_index})")

class BlobObservation:
    def __init__(self):
        self.blobs = []
        self.num_blobs = 0
        self.dropped_dark_blobs = 0

class BlobWatch:
    def __init__(self, 
                 pixel_threshold=50, 
                 min_area=5, 
                 max_area=1e5, 
                 history_length=100,
                 pyramid_scales=None,
                 max_merge_distance=10.0):
        """
        :param pixel_threshold: Threshold for binarizing the image
        :param min_area: Minimum area for a valid blob
        :param max_area: Maximum area for a valid blob
        :param history_length: Number of previous observations to buffer
        :param pyramid_scales: List of downscale factors for the pyramid 
                               (e.g. [1.0, 0.5, 0.25]). 1.0 means original size.
        :param max_merge_distance: Distance threshold for merging duplicate 
                                   blob detections across scales
        """
        self.pixel_threshold = pixel_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_match_distance = 50.0
        self.next_blob_id = 0
        self.next_track_index = 0
        self.buffer_capacity = history_length + 1
        self.observation_buffer = deque()
        for _ in range(history_length):
            self.observation_buffer.append(BlobObservation())
        self.previous_observation = None
        
        self.pyramid_scales = pyramid_scales if pyramid_scales else [1.0, 0.5]
        self.max_merge_distance = max_merge_distance

    def process_frame(self, frame):

        pyramid_frames = []
        for scale in self.pyramid_scales:
            if scale == 1.0:
                scaled_frame = frame
            else:
                new_w = int(frame.shape[1] * scale)
                new_h = int(frame.shape[0] * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pyramid_frames.append((scale, scaled_frame))

        all_blobs = []
        for scale, sframe in pyramid_frames:
            scale_blobs = self._detect_blobs_in_frame(sframe, scale)
            all_blobs.extend(scale_blobs)

        merged_blobs = self._merge_candidate_blobs(all_blobs, self.max_merge_distance)

        if self.observation_buffer:
            observation = self.observation_buffer.popleft()
            observation.blobs.clear()
            observation.num_blobs = 0
            observation.dropped_dark_blobs = 0
        else:
            observation = BlobObservation()

        observation.blobs = merged_blobs
        observation.num_blobs = len(merged_blobs)

        self._match_and_update_ids(observation)

        self.previous_observation = observation

        return observation

    def _detect_blobs_in_frame(self, frame, scale=1.0):
        """
        Runs threshold + morphological ops + connectedComponents
        on the given (scaled) frame, returning a list of Blob objects
        whose coordinates are converted back to the original scale.
        """

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Threshold
        _, binary = cv2.threshold(gray, self.pixel_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        detected_blobs = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            # Filter by area at the scaled level
            if area < self.min_area or area > self.max_area:
                continue

            left = stats[label_id, cv2.CC_STAT_LEFT]
            top = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]

            cx, cy = centroids[label_id]
            if scale != 1.0:
                cx_orig = cx / scale
                cy_orig = cy / scale
                left_orig = left / scale
                top_orig = top / scale
                width_orig = width / scale
                height_orig = height / scale
            else:
                cx_orig = cx
                cy_orig = cy
                left_orig = float(left)
                top_orig = float(top)
                width_orig = float(width)
                height_orig = float(height)

            blob = Blob(blob_id=-1,
                        x=cx_orig,
                        y=cy_orig,
                        area=area,  
                        top=int(top_orig),
                        left=int(left_orig),
                        width=int(width_orig),
                        height=int(height_orig))
            detected_blobs.append(blob)

        return detected_blobs

    def _merge_candidate_blobs(self, blobs, merge_distance):
        merged = []
        visited = [False] * len(blobs)

        for i in range(len(blobs)):
            if visited[i]:
                continue

            cluster_indices = [i]
            queue = [i]
            visited[i] = True

            while queue:
                current_idx = queue.pop(0)
                for j in range(len(blobs)):
                    if not visited[j]:
                        if self._blob_distance(blobs[current_idx], blobs[j]) < merge_distance:
                            visited[j] = True
                            queue.append(j)
                            cluster_indices.append(j)

            cluster_blobs = [blobs[idx] for idx in cluster_indices]
            merged_blob = self._combine_blob_cluster(cluster_blobs)
            merged.append(merged_blob)

        return merged

    def _blob_distance(self, b1, b2):

        dx = b1.x - b2.x
        dy = b1.y - b2.y
        return sqrt(dx*dx + dy*dy)

    def _combine_blob_cluster(self, cluster_blobs):

        if not cluster_blobs:
            return None
        if len(cluster_blobs) == 1:
            return cluster_blobs[0]

        total_area = sum(b.area for b in cluster_blobs)
        if total_area < 1e-6:
            total_area = len(cluster_blobs)

        x_sum = 0.0
        y_sum = 0.0
        for b in cluster_blobs:
            x_sum += b.x * b.area
            y_sum += b.y * b.area

        avg_x = x_sum / total_area
        avg_y = y_sum / total_area

        lefts   = [b.left for b in cluster_blobs]
        tops    = [b.top for b in cluster_blobs]
        rights  = [b.left + b.width for b in cluster_blobs]
        bottoms = [b.top + b.height for b in cluster_blobs]

        union_left   = min(lefts)
        union_top    = min(tops)
        union_right  = max(rights)
        union_bottom = max(bottoms)

        new_blob = Blob(
            blob_id=-1, 
            x=avg_x,
            y=avg_y,
            area=int(sum(b.area for b in cluster_blobs)),
            top=union_top,
            left=union_left,
            width=union_right - union_left,
            height=union_bottom - union_top,
        )
        return new_blob

    def _match_and_update_ids(self, observation):
        """
        Matches newly detected blobs with old ones based on proximity
        to predicted positions (x+vx, y+vy). If a match is found
        within self.max_match_distance, we reuse the old ID; otherwise
        we assign a new ID.
        """
        new_blobs = observation.blobs
        old_ob = self.previous_observation
        if old_ob is None:
            for blob in new_blobs:
                blob.blob_id = self.next_blob_id
                self.next_blob_id += 1
                blob.track_index = self.next_track_index
                self.next_track_index += 1
            return

        old_blobs = old_ob.blobs
        predicted = [(old.x + old.vx, old.y + old.vy) for old in old_blobs]
        distances = {}

        for i, new_blob in enumerate(new_blobs):
            for j, (pred_x, pred_y) in enumerate(predicted):
                dx = new_blob.x - pred_x
                dy = new_blob.y - pred_y
                dist = (dx*dx + dy*dy) ** 0.5
                if dist < self.max_match_distance:
                    distances[(i, j)] = dist

        new_matches = [-1] * len(new_blobs)
        old_matches = [-1] * len(old_blobs)

        for (i, j), dist in sorted(distances.items(), key=lambda item: item[1]):
            if new_matches[i] == -1 and old_matches[j] == -1:
                new_matches[i] = j
                old_matches[j] = i

        for i, new_blob in enumerate(new_blobs):
            if new_matches[i] != -1:
                old_blob = old_blobs[new_matches[i]]
                new_blob.blob_id = old_blob.blob_id
                new_blob.vx = new_blob.x - old_blob.x
                new_blob.vy = new_blob.y - old_blob.y
                new_blob.age = old_blob.age + 1
                new_blob.track_index = old_blob.track_index
            else:
                new_blob.blob_id = self.next_blob_id
                self.next_blob_id += 1
                new_blob.track_index = self.next_track_index
                self.next_track_index += 1

    def find_blob_at(self, x, y, tolerance=5.0):
        """
        Find a blob in the last observation near (x,y) within 'tolerance'
        """
        if self.previous_observation is None:
            return None
        for blob in self.previous_observation.blobs:
            dx = blob.x - x
            dy = blob.y - y
            if (dx*dx + dy*dy) ** 0.5 <= tolerance:
                return blob
        return None

    def release_observation(self, observation):

        self.observation_buffer.append(observation)
