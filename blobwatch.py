import cv2
import numpy as np
import logging
from collections import deque

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
    def __init__(self, pixel_threshold=50, min_area=5, max_area=1e5, history_length=5):
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

    def process_frame(self, frame):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        _, binary = cv2.threshold(gray, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if self.observation_buffer:
            observation = self.observation_buffer.popleft()
            observation.blobs.clear()
            observation.num_blobs = 0
            observation.dropped_dark_blobs = 0
        else:
            observation = BlobObservation()

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < self.min_area or area > self.max_area:
                observation.dropped_dark_blobs += 1
                continue

            left = stats[label_id, cv2.CC_STAT_LEFT]
            top = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]
            region = gray[top:top+height, left:left+width]
            m = cv2.moments(region)
            if m["m00"] != 0:
                cx = (m["m10"] / m["m00"]) + left - 1
                cy = (m["m01"] / m["m00"]) + top - 1
            else:
                cx, cy = centroids[label_id]

            blob = Blob(blob_id=-1, x=cx, y=cy, area=area, top=top, left=left, width=width, height=height)
            observation.blobs.append(blob)

        observation.num_blobs = len(observation.blobs)
        self._match_and_update_ids(observation)
        self.previous_observation = observation
        return observation

    def _match_and_update_ids(self, observation):
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
