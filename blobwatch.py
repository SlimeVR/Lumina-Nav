import numpy as np
from numba import jit, njit, types, typed
from numba.typed import List
from dataclasses import dataclass
import cv2
from typing import Optional, Tuple

NUM_FRAMES_HISTORY = 5
MAX_EXTENTS_PER_LINE = 30
QUEUE_ENTRIES = NUM_FRAMES_HISTORY + 1
MAX_BLOBS_PER_FRAME = 100
LED_INVALID_ID = 0xFFFF

extent_dtype = np.dtype([
    ('start', np.uint16),
    ('end', np.uint16),
    ('top', np.uint16),
    ('left', np.uint16),
    ('right', np.uint16),
    ('area', np.uint32),
    ('max_pixel', np.uint8)
])

blob_dtype = np.dtype([
    ('blob_id', np.uint32),
    ('x', np.float32),
    ('y', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('left', np.uint16),
    ('top', np.uint16),
    ('width', np.uint16),
    ('height', np.uint16),
    ('area', np.uint32),
    ('age', np.uint16),
    ('track_index', np.int16),
    ('id_age', np.uint16),
    ('prev_led_id', np.uint16),
    ('led_id', np.uint16),
    ('brightness', np.uint8)
])

@njit
def min_val(x, y):
    return x if x < y else y

@njit
def max_val(x, y):
    return x if x > y else y

@njit
def abs_val(x):
    return x if x >= 0 else -x

class ExtentLine:
    def __init__(self):
        self.extents = np.zeros(MAX_EXTENTS_PER_LINE, dtype=extent_dtype)
        self.num = 0

class Blobservation:
    def __init__(self):
        self.blobs = np.zeros(MAX_BLOBS_PER_FRAME, dtype=blob_dtype)
        self.num_blobs = 0
        self.tracked = np.zeros(MAX_BLOBS_PER_FRAME, dtype=np.uint8)
        self.dropped_dark_blobs = 0

class BlobservationQueue:
    def __init__(self, observations):
        self.data = [None] * QUEUE_ENTRIES
        self.head = 0
        self.tail = 0
        #initialize with observations
        for obs in observations:
            self.push(obs)
    
    def push(self, observation):
        next_tail = (self.tail + 1) % QUEUE_ENTRIES
        assert next_tail != self.head, "Queue full"
        self.data[self.tail] = observation
        self.tail = next_tail
    
    def pop(self):
        if self.tail == self.head:
            return None
        observation = self.data[self.head]
        self.head = (self.head + 1) % QUEUE_ENTRIES
        return observation

class Blobwatch:
    def __init__(self, pixel_threshold=10, blob_required_threshold=20):
        self.next_blob_id = 1
        self.pixel_threshold = pixel_threshold
        self.blob_required_threshold = blob_required_threshold
        self.blob_max_wh = 35
        self.debug = True
        
        self.observations = [Blobservation() for _ in range(NUM_FRAMES_HISTORY)]
        self.observation_q = BlobservationQueue(self.observations)
        self.last_observation = None

@njit
def compute_greysum(frame: np.ndarray, extent: np.ndarray, end_y: int) -> Tuple[float, float]:
    left = extent['left']
    right = extent['right']
    top = extent['top']
    
    width = right - left + 1
    height = end_y - top + 1
    
    greysum_total = 0
    greysum_x = 0
    greysum_y = 0
    
    for y in range(height):
        y_pos = top + y + 1
        x_pos = left + 1
        
        for x in range(width):
            pix = frame[top + y, left + x]
            greysum_total += pix
            greysum_x += x_pos * pix
            greysum_y += y_pos * pix
            x_pos += 1
    
    if greysum_total > 0:
        led_x = float(greysum_x) / greysum_total - 1
        led_y = float(greysum_y) / greysum_total - 1
    else:
        led_x = float(left + right) / 2
        led_y = float(top + end_y) / 2
    
    return led_x, led_y

@njit
def store_blob(extent: np.ndarray, index: int, end_y: int, blobs: np.ndarray,
               blob_id: int, led_x: float, led_y: float, brightness: int):
    b = blobs[index]
    b['blob_id'] = blob_id
    b['x'] = led_x
    b['y'] = led_y
    b['vx'] = 0
    b['vy'] = 0
    b['left'] = extent['left']
    b['top'] = extent['top']
    b['width'] = extent['right'] - extent['left'] + 1
    b['height'] = end_y - extent['top'] + 1
    b['area'] = extent['area']
    b['age'] = 0
    b['track_index'] = -1
    b['id_age'] = 0
    b['prev_led_id'] = LED_INVALID_ID
    b['led_id'] = LED_INVALID_ID
    b['brightness'] = brightness

def extent_to_blobs(bw: Blobwatch, ob: Blobservation, extent: np.ndarray, 
                    y: int, frame: np.ndarray):
    
    if extent['max_pixel'] < bw.blob_required_threshold:
        ob.dropped_dark_blobs += 1
        return
    
    if extent['top'] == y and extent['left'] == extent['right']:
        return
    
    if y - extent['top'] > bw.blob_max_wh or extent['right'] - extent['left'] > bw.blob_max_wh:
        return
    
    if ob.num_blobs < MAX_BLOBS_PER_FRAME:
        led_x, led_y = compute_greysum(frame, extent, y)
        store_blob(extent, ob.num_blobs, y, ob.blobs, 
                  bw.next_blob_id, led_x, led_y, extent['max_pixel'])
        bw.next_blob_id += 1
        ob.num_blobs += 1

@njit
def process_scanline_core(line: np.ndarray, pixel_threshold: int, y: int,
                          el_extents: np.ndarray, el_num: int,
                          prev_el_extents: np.ndarray, prev_el_num: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """Core scanline processing with Numba optimization"""
    
    width = len(line)
    finished_extents = np.zeros(MAX_EXTENTS_PER_LINE, dtype=extent_dtype)
    num_finished = 0
    
    le_idx = 0 
    e = 0
    
    x = 0
    while x < width and e < MAX_EXTENTS_PER_LINE:
        if line[x] <= pixel_threshold:
            x += 1
            continue
        
        start = x
        max_pixel = line[x]
        x += 1
        
        while x < width and line[x] > pixel_threshold:
            if line[x] > max_pixel:
                max_pixel = line[x]
            x += 1
        
        end = x - 1
        center = (start + end) / 2.0
        
        extent = el_extents[e]
        extent['start'] = start
        extent['end'] = end
        extent['area'] = x - start
        extent['max_pixel'] = max_pixel
        
        is_new_extent = True
        
        if prev_el_num > 0:
            while le_idx < prev_el_num and prev_el_extents[le_idx]['end'] < center:
                finished_extents[num_finished] = prev_el_extents[le_idx]
                num_finished += 1
                le_idx += 1
            
            if le_idx < prev_el_num:
                le = prev_el_extents[le_idx]
                if le['start'] <= center and le['end'] >= center:
                    extent['top'] = le['top']
                    extent['left'] = min_val(extent['start'], le['left'])
                    extent['right'] = max_val(extent['end'], le['right'])
                    if le['max_pixel'] > extent['max_pixel']:
                        extent['max_pixel'] = le['max_pixel']
                    extent['area'] += le['area']
                    is_new_extent = False
                    le_idx += 1
        
        if is_new_extent:
            extent['top'] = y
            extent['left'] = extent['start']
            extent['right'] = extent['end']
        
        e += 1
    
    while le_idx < prev_el_num:
        finished_extents[num_finished] = prev_el_extents[le_idx]
        num_finished += 1
        le_idx += 1
    
    return el_extents, e, finished_extents[:num_finished]

def process_scanline(line: np.ndarray, bw: Blobwatch, y: int, 
                    el: ExtentLine, prev_el: Optional[ExtentLine],
                    frame: np.ndarray, ob: Blobservation):
    """Process a single scanline to find extents"""
    
    prev_extents = prev_el.extents if prev_el else np.zeros(0, dtype=extent_dtype)
    prev_num = prev_el.num if prev_el else 0
    
    _, el.num, finished_extents = process_scanline_core(
        line, bw.pixel_threshold, y, el.extents, el.num,
        prev_extents, prev_num
    )
    
    for extent in finished_extents:
        extent_to_blobs(bw, ob, extent, y, frame)
    
    if y == frame.shape[0] - 1:
        for i in range(el.num):
            extent_to_blobs(bw, ob, el.extents[i], y, frame)

def process_frame(bw: Blobwatch, ob: Blobservation, frame: np.ndarray):
    ob.num_blobs = 0
    ob.dropped_dark_blobs = 0
    
    el1 = ExtentLine()
    el2 = ExtentLine()
    
    process_scanline(frame[0], bw, 0, el1, None, frame, ob)
    
    for y in range(1, frame.shape[0]):
        if y & 1:
            process_scanline(frame[y], bw, y, el2, el1, frame, ob)
        else:
            process_scanline(frame[y], bw, y, el1, el2, frame, ob)

@njit
def find_free_track(tracked: np.ndarray) -> int:
    for i in range(len(tracked)):
        if tracked[i] == 0:
            return i
    return -1

@njit
def copy_matching_blob(to_blob: np.ndarray, from_blob: np.ndarray):
    to_blob['blob_id'] = from_blob['blob_id']
    to_blob['vx'] = to_blob['x'] - from_blob['x']
    to_blob['vy'] = to_blob['y'] - from_blob['y']
    to_blob['id_age'] = from_blob['id_age']
    to_blob['led_id'] = from_blob['led_id']
    to_blob['age'] = from_blob['age'] + 1

def blobwatch_process(bw: Blobwatch, frame: np.ndarray) -> Optional[Blobservation]:
    
    ob = bw.observation_q.pop()
    if ob is None:
        return None
    
    process_frame(bw, ob, frame)
    
    if bw.last_observation is None:
        bw.last_observation = ob
        return ob
    
    last_ob = bw.last_observation
    
    closest_ob = np.full(MAX_BLOBS_PER_FRAME, -1, dtype=np.int32)
    closest_last_ob = np.full(MAX_BLOBS_PER_FRAME, -1, dtype=np.int32)
    closest_last_ob_distsq = np.full(MAX_BLOBS_PER_FRAME, 1000000, dtype=np.int32)
    
    scan_again = 1
    scan_times = 0
    
    while scan_again:
        scan_again = 0
        
        for i in range(ob.num_blobs):
            if closest_ob[i] != -1:
                continue
            
            b2 = ob.blobs[i]
            closest_j = -1
            closest_distsq = -1
            
            for j in range(last_ob.num_blobs):
                b1 = last_ob.blobs[j]
                
                x = b1['x'] + b1['vx']
                y = b1['y'] + b1['vy']
                
                dx = abs_val(x - b2['x'])
                dy = abs_val(y - b2['y'])
                distsq = dx * dx + dy * dy
                
                if closest_distsq < 0 or distsq < closest_distsq:
                    if closest_last_ob[j] != -1 and closest_last_ob_distsq[j] <= distsq:
                        continue
                    closest_j = j
                    closest_distsq = distsq
            
            closest_ob[i] = closest_j
            
            if closest_j < 0:
                continue
            
            if closest_last_ob[closest_j] != -1:
                closest_ob[closest_last_ob[closest_j]] = -1
                scan_again += 1
            
            closest_last_ob[closest_j] = i
            closest_last_ob_distsq[closest_j] = closest_distsq
        
        scan_times += 1
        if scan_times > 100:
            print(f"Warning: blob matching looped excessively. scan_times: {scan_times}")
            break
    
    for i in range(ob.num_blobs):
        if closest_ob[i] < 0:
            continue
        
        b2 = ob.blobs[i]
        b1 = last_ob.blobs[closest_ob[i]]
        
        if b1['track_index'] >= 0 and ob.tracked[b1['track_index']] == 0:
            b2['track_index'] = b1['track_index']
            ob.tracked[b2['track_index']] = i + 1
        
        copy_matching_blob(b2, b1)
    
    for i in range(MAX_BLOBS_PER_FRAME):
        t = ob.tracked[i]
        if t > 0 and ob.blobs[t - 1]['track_index'] != i:
            ob.tracked[i] = 0
    
    for i in range(ob.num_blobs):
        b2 = ob.blobs[i]
        if b2['age'] > 0 and b2['track_index'] < 0:
            b2['track_index'] = find_free_track(ob.tracked)
        if b2['track_index'] >= 0:
            ob.tracked[b2['track_index']] = i + 1
    
    bw.last_observation = ob
    return ob

def blobwatch_find_blob_at(bw: Blobwatch, x: int, y: int) -> Optional[np.ndarray]:
    if bw.last_observation is None:
        return None
    
    ob = bw.last_observation
    for i in range(ob.num_blobs):
        b = ob.blobs[i]
        dx = abs_val(x - b['x'])
        dy = abs_val(y - b['y'])
        
        if 2 * dx <= b['width'] and 2 * dy <= b['height']:
            return b
    
    return None

def blobwatch_update_labels(bw: Blobwatch, ob: Blobservation, device_id: int):
    last_ob = bw.last_observation
    
    if last_ob is None or last_ob == ob:
        for i in range(ob.num_blobs):
            b = ob.blobs[i]
            if b['led_id'] != LED_INVALID_ID and b['led_id'] == b['prev_led_id']:
                b['id_age'] += 1
            else:
                b['id_age'] = 0
        return
    
    for l in range(last_ob.num_blobs):
        new_b = last_ob.blobs[l]
        if (new_b['led_id'] >> 8) == device_id:
            new_b['prev_led_id'] = new_b['led_id']
            new_b['led_id'] = LED_INVALID_ID
    
    for i in range(ob.num_blobs):
        b = ob.blobs[i]
        if (b['led_id'] >> 8) != device_id:
            continue
        
        for l in range(last_ob.num_blobs):
            new_b = last_ob.blobs[l]
            if new_b['blob_id'] == b['blob_id']:
                if bw.debug:
                    print(f"Found matching blob {b['blob_id']} with labelled with LED id {b['led_id']:x}")
                new_b['led_id'] = b['led_id']
                
                if new_b['led_id'] == new_b['prev_led_id']:
                    new_b['id_age'] += 1
                else:
                    new_b['id_age'] = 0

def blobwatch_release_observation(bw: Blobwatch, ob: Blobservation):
    bw.observation_q.push(ob)


# Example usage
if __name__ == "__main__":
    bw = Blobwatch(pixel_threshold=10, blob_required_threshold=20)

    frame = np.zeros((480, 640), dtype=np.uint8)
    frame[100:110, 100:110] = 100
    frame[200:215, 300:315] = 150
    frame[350:360, 500:510] = 200
    
    observation = blobwatch_process(bw, frame)
    
    if observation:
        print(f"Found {observation.num_blobs} blobs")
        for i in range(observation.num_blobs):
            blob = observation.blobs[i]
            print(f"Blob {i}: id={blob['blob_id']}, pos=({blob['x']:.2f}, {blob['y']:.2f}), "
                  f"size={blob['width']}x{blob['height']}, brightness={blob['brightness']}")
        
        blobwatch_release_observation(bw, observation)