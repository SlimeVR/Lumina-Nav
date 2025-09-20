from __future__ import annotations
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import cv2

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco
    prange = range
    NUMBA_AVAILABLE = False

MAX_LED_SEARCH_DEPTH = 8
MAX_BLOB_SEARCH_DEPTH = 16

POSE_MATCH_GOOD = 1 << 0
POSE_MATCH_STRONG = 1 << 1

CS_FLAG_SHALLOW_SEARCH      = 1 << 0
CS_FLAG_DEEP_SEARCH         = 1 << 1
CS_FLAG_MATCH_ALL_BLOBS     = 1 << 2
CS_FLAG_STOP_FOR_STRONG     = 1 << 3
CS_FLAG_HAVE_POSE_PRIOR     = 1 << 4
CS_FLAG_MATCH_GRAVITY       = 1 << 5

LED_INVALID_ID = 0xFFFF

MIN_TRI_AREA_NORM = 1.5e-4   #minimum normalized image triangle area for P3P sampling
MIN_PAIR_SEP_PX   = 2.0      #min pixel separation between any pair of P3P sample points

@njit(cache=True, fastmath=True)
def _quat_from_R_numba(R: np.ndarray) -> np.ndarray:
    t = R[0, 0] + R[1, 1] + R[2, 2]
    q = np.empty(4, dtype=np.float64)
    if t > 0.0:
        s = 0.5 / math.sqrt(t + 1.0)
        q[3] = -0.25 / s
        q[0] = -(R[2, 1] - R[1, 2]) * s
        q[1] = -(R[0, 2] - R[2, 0]) * s
        q[2] = -(R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = -(R[2, 1] - R[1, 2]) / s
            q[0] = -0.25 * s
            q[1] = -(R[0, 1] + R[1, 0]) / s
            q[2] = -(R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = -(R[0, 2] - R[2, 0]) / s
            q[0] = -(R[0, 1] + R[1, 0]) / s
            q[1] = -0.25 * s
            q[2] = -(R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = -(R[1, 0] - R[0, 1]) / s
            q[0] = -(R[0, 2] + R[2, 0]) / s
            q[1] = -(R[1, 2] + R[2, 1]) / s
            q[2] = -0.25 * s
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if n > 0.0:
        q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n
    return q

def quat_from_R(R: np.ndarray) -> np.ndarray:
    return _quat_from_R_numba(R.astype(np.float64))

@njit(cache=True, fastmath=True, parallel=True)
def _sorted_sqdist_indices(coords: np.ndarray) -> np.ndarray:
    N = coords.shape[0]
    out = np.empty((N, N-1), np.int32)
    for i in prange(N):
        xi = coords[i, 0]; yi = coords[i, 1]
        d2 = np.empty(N-1, np.float64)
        idx = np.empty(N-1, np.int32)
        c = 0
        for j in range(N):
            if j == i:
                continue
            dx = coords[j, 0] - xi
            dy = coords[j, 1] - yi
            d2[c] = dx*dx + dy*dy
            idx[c] = j
            c += 1
        order = np.argsort(d2)
        out[i, :] = idx[order]
    return out

@dataclass
class Blob:
    x: float
    y: float
    width: float
    height: float
    led_id: int = LED_INVALID_ID

@dataclass
class ImagePoint:
    point_homog: np.ndarray  #undistorted normalized
    size: np.ndarray         #normalized
    max_dist: float
    blob: Blob
    neighbours: List[int]

@dataclass
class LED:
    pos: np.ndarray  #(3,)
    dir: np.ndarray  #(3,)
    id: int

@dataclass
class LEDNeighborList:
    led_index: int
    neighbours: List[int]

@dataclass
class PoseMetrics:
    matched_blobs: int = 0
    visible_leds: int = 0
    unmatched_blobs: int = 0
    reprojection_error: float = 0.0
    match_flags: int = 0

class CorrespondenceSearch:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        led_positions: np.ndarray,
        led_normals: np.ndarray,
        led_ids: Optional[np.ndarray] = None,
        model_id: int = 0,
        max_led_search_depth: int = MAX_LED_SEARCH_DEPTH,
        max_blob_search_depth: int = MAX_BLOB_SEARCH_DEPTH,
        parallel_anchors: bool = False,
        max_cv_threads: Optional[int] = None,
    ) -> None:
        self.K = np.ascontiguousarray(camera_matrix, dtype=np.float64)
        self.dist = np.ascontiguousarray(dist_coeffs.reshape(-1), dtype=np.float64)
        self.fx = float(self.K[0, 0]); self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2]); self.cy = float(self.K[1, 2])

        if max_cv_threads is not None:
            try:
                cv2.setNumThreads(int(max_cv_threads))
            except Exception:
                pass

        led_positions = np.asarray(led_positions, dtype=np.float64)
        led_normals   = np.asarray(led_normals,   dtype=np.float64)

        if led_positions.ndim != 2 or led_positions.shape[0] == 0:
            raise ValueError(f"led_positions must be (N,3); got shape {led_positions.shape}")
        if led_positions.shape[1] < 3:
            raise ValueError(f"led_positions must have 3 columns; got {led_positions.shape[1]}")
        if led_positions.shape[1] > 3:
            warnings.warn(f"led_positions has {led_positions.shape[1]} columns; using the first 3.", RuntimeWarning)
            led_positions = led_positions[:, :3]

        if led_normals.ndim == 1:
            if led_normals.size == 3:
                led_normals = np.tile(led_normals, (led_positions.shape[0], 1))
            else:
                raise ValueError(f"led_normals must be (N,3) or (3,); got length {led_normals.size}")
        elif led_normals.ndim == 2:
            if led_normals.shape[0] != led_positions.shape[0]:
                raise ValueError(f"led_normals rows ({led_normals.shape[0]}) must match led_positions rows ({led_positions.shape[0]}).")
            if led_normals.shape[1] < 3:
                raise ValueError(f"led_normals must have at least 3 columns; got {led_normals.shape[1]}")
            if led_normals.shape[1] > 3:
                warnings.warn(f"led_normals has {led_normals.shape[1]} columns; using the first 3.", RuntimeWarning)
                led_normals = led_normals[:, :3]
        else:
            raise ValueError(f"led_normals must be (N,3) or (3,); got shape {led_normals.shape}")

        led_positions = np.ascontiguousarray(led_positions, dtype=np.float64)
        led_normals   = np.ascontiguousarray(led_normals,   dtype=np.float64)

        self.model_id = int(model_id)
        self.leds: List[LED] = []
        for i in range(led_positions.shape[0]):
            pos = led_positions[i]
            nrm = led_normals[i]
            nrm = nrm[:3] / (np.linalg.norm(nrm[:3]) + 1e-12)
            lid = int(led_ids[i]) if led_ids is not None else i
            self.leds.append(LED(pos=pos.copy(), dir=nrm.copy(), id=lid))
        self.N_leds = len(self.leds)

        self._led_pos_T = np.ascontiguousarray(np.stack([ld.pos for ld in self.leds], axis=1), dtype=np.float64)
        self._led_dir_T = np.ascontiguousarray(np.stack([ld.dir for ld in self.leds], axis=1), dtype=np.float64)

        self.max_led_search_depth = int(max_led_search_depth)
        self.max_blob_search_depth = int(max_blob_search_depth)
        self.led_neighbors: List[LEDNeighborList] = self._build_led_neighbors()

        self._blobs: List[Blob] = []
        self._points: List[ImagePoint] = []
        self._blob_neighbours: List[List[int]] = []

        self.z_min = 0.05
        self.z_max = 25.0
        self.backcheck_tol_norm = 4.0e-3
        self.strong_min_inliers = 11
        self.strong_max_rmse_px = 2.0
        self.gate_px_scale = 0.7

        self.parallel_anchors = bool(parallel_anchors)
        self.max_workers = min(8, (os.cpu_count() or 4))

        self._eye3 = np.eye(3, dtype=np.float64)
        self._p3p_obj = np.empty((3, 3), np.float64)
        self._p3p_img = np.empty((3, 2), np.float64)
        self._proj_buf = None  #will be (3, N_leds)


    def set_blobs(self, blobs: List[Blob], search_flags: int = 0) -> None:
        self._blobs = list(blobs)
        N = len(blobs)
        if N == 0:
            self._points = []
            self._blob_neighbours = []
            return

        pix = np.ascontiguousarray([[b.x, b.y] for b in blobs], dtype=np.float64).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(pix, self.K, self.dist).reshape(-1, 2)

        points: List[ImagePoint] = []
        for i, b in enumerate(blobs):
            ph = np.array([undist[i, 0], undist[i, 1], 1.0], dtype=np.float64)
            sx = b.width / self.fx
            sy = b.height / self.fy
            md = float(math.hypot(sx, sy))
            points.append(ImagePoint(point_homog=ph, size=np.array([sx, sy], dtype=np.float64), max_dist=md, blob=b, neighbours=[]))
        self._points = points

        coords = np.ascontiguousarray([[b.x, b.y] for b in blobs], dtype=np.float64)
        all_sorted = _sorted_sqdist_indices(coords) if NUMBA_AVAILABLE and N >= 3 else None
        neigh_lists: List[List[int]] = []
        for i in range(N):
            order = list(all_sorted[i]) if all_sorted is not None else np.argsort(np.sum((coords - coords[i])**2, axis=1)).tolist()
            order = [int(j) for j in order if j != i]

            K = self.max_blob_search_depth
            filtered: List[int] = []

            if (search_flags & CS_FLAG_MATCH_ALL_BLOBS) != 0:
                if len(order) <= K:
                    filtered = order
                else:
                    half = K // 2
                    filtered = order[:half] + order[-half:]
            else:
                #respect led_id if present, but still prefer spread: alternate near/far
                lo, hi = 0, len(order) - 1
                while len(filtered) < min(K, len(order)) and lo <= hi:
                    for idx in (lo, hi):
                        if 0 <= idx < len(order):
                            j = order[idx]
                            lid = self._blobs[j].led_id
                            if lid == LED_INVALID_ID or self._same_model(lid):
                                if j not in filtered:
                                    filtered.append(j)
                                    if len(filtered) >= K:
                                        break
                    lo += 1; hi -= 1

            neigh_lists.append(filtered)
        self._blob_neighbours = neigh_lists
        for i, ip in enumerate(self._points):
            ip.neighbours = neigh_lists[i]

    def solve(
        self,
        search_flags: int = (CS_FLAG_SHALLOW_SEARCH | CS_FLAG_DEEP_SEARCH | CS_FLAG_STOP_FOR_STRONG),
        pose_prior: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        pos_error_thresh: Optional[np.ndarray] = None,
        rot_error_thresh: Optional[np.ndarray] = None,
        gravity_vector: Optional[np.ndarray] = None,
        gravity_tolerance_rad: float = math.radians(15.0),
    ) -> Tuple[bool, Dict[str, Any]]:
        if len(self._points) == 0:
            return False, {"reason": "no blobs"}

        if (search_flags & (CS_FLAG_SHALLOW_SEARCH | CS_FLAG_DEEP_SEARCH)) == 0:
            search_flags |= CS_FLAG_SHALLOW_SEARCH | CS_FLAG_DEEP_SEARCH

        if (search_flags & CS_FLAG_SHALLOW_SEARCH) != 0:
            max_blob_depth = min(4, self.max_blob_search_depth)
            max_led_depth = min(4, self.max_led_search_depth)
            min_led_depth = 1
        else:
            max_blob_depth = self.max_blob_search_depth
            max_led_depth = self.max_led_search_depth
            min_led_depth = 3
        if (search_flags & CS_FLAG_DEEP_SEARCH) != 0:
            max_blob_depth = self.max_blob_search_depth
            max_led_depth = self.max_led_search_depth

        use_pose_prior = (search_flags & CS_FLAG_HAVE_POSE_PRIOR) != 0 and pose_prior is not None
        use_gravity = (search_flags & CS_FLAG_MATCH_GRAVITY) != 0 and gravity_vector is not None and use_pose_prior
        if use_gravity:
            g = gravity_vector.astype(np.float64)
            g = g / (np.linalg.norm(g) + 1e-12)
            R0, _t0 = self._parse_pose_prior(pose_prior)
            swing_prior, _ = self._quat_decompose_swing_twist(quat_from_R(R0), g)
        else:
            g = None
            swing_prior = None

        best_metrics = None
        best_R = None
        best_t = None
        best_blob_depth = -1
        best_led_depth = -1
        num_trials = 0
        num_pose_checks = 0

        #iterate LED anchors
        for l_idx, neigh in enumerate(self.led_neighbors):
            candidates = neigh.neighbours
            if len(candidates) < 3:
                continue
            start = min_led_depth - 1
            end = min(len(candidates), max_led_depth)
            led_slice = candidates[start:end]
            if len(led_slice) < 3:
                continue
            L = len(led_slice)
            for i1 in range(L):
                for i2 in range(i1+1, L):
                    for i3 in range(i2+1, L):
                        ml = [l_idx, led_slice[i1], led_slice[i2], led_slice[i3]]
                        for perm in ((0,1,2,3), (0,2,1,3)):
                            model_leds = [ml[p] for p in perm]
                            improved, out = self._check_led_match(
                                model_leds, max_blob_depth,
                                use_pose_prior, pose_prior,
                                pos_error_thresh, rot_error_thresh,
                                use_gravity, g, swing_prior,
                                gravity_tolerance_rad,
                            )
                            num_trials += out.get('num_trials', 0)
                            num_pose_checks += out.get('num_pose_checks', 0)

                            cand_metrics = out.get('metrics', None)
                            if improved and cand_metrics is not None:
                                if (best_metrics is None) or self._is_better_pose(best_metrics, cand_metrics):
                                    best_metrics = cand_metrics
                                    best_R = out['R']; best_t = out['t']
                                    best_led_depth = out['led_depth']; best_blob_depth = out['blob_depth']

                                    if (best_metrics.match_flags & POSE_MATCH_STRONG) and (search_flags & CS_FLAG_STOP_FOR_STRONG):
                                        ok = (best_metrics.match_flags & POSE_MATCH_GOOD) != 0
                                        return ok, {
                                            'R': best_R, 't': best_t, 'q': quat_from_R(best_R),
                                            'metrics': best_metrics,
                                            'num_trials': num_trials,
                                            'num_pose_checks': num_pose_checks,
                                            'best_led_depth': best_led_depth,
                                            'best_blob_depth': best_blob_depth,
                                        }

        ok = best_metrics is not None and (best_metrics.match_flags & POSE_MATCH_GOOD) != 0
        if ok:
            return True, {
                'R': best_R, 't': best_t, 'q': quat_from_R(best_R),
                'metrics': best_metrics,
                'num_trials': num_trials,
                'num_pose_checks': num_pose_checks,
                'best_led_depth': best_led_depth,
                'best_blob_depth': best_blob_depth,
            }
        else:
            return False, {'reason': 'no good pose', 'metrics': best_metrics if best_metrics else PoseMetrics(), 'num_trials': num_trials, 'num_pose_checks': num_pose_checks}
    

    def _same_model(self, led_id: int) -> bool:
        return (led_id >> 16) == self.model_id

    def _build_led_neighbors(self) -> List[LEDNeighborList]:
        pos = np.stack([ld.pos for ld in self.leds], axis=0)
        dirv = np.stack([ld.dir for ld in self.leds], axis=0)
        out: List[LEDNeighborList] = []
        for i in range(self.N_leds):
            dots = dirv @ dirv[i]
            ok = dots >= 0.0
            ok[i] = False
            d2 = np.sum((pos - pos[i])**2, axis=1)
            idx = np.where(ok)[0]
            idx_sorted = idx[np.argsort(d2[idx])]
            out.append(LEDNeighborList(led_index=i, neighbours=list(map(int, idx_sorted[:self.max_led_search_depth]))))
        return out

    def _parse_pose_prior(self, pose_prior: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        R_or_rvec, t = pose_prior
        t = np.asarray(t, dtype=np.float64).reshape(3)
        R_or_rvec = np.asarray(R_or_rvec, dtype=np.float64)
        if R_or_rvec.shape == (3, 3):
            R = R_or_rvec
        else:
            R, _ = cv2.Rodrigues(R_or_rvec.reshape(3,))
        return R, t

    def _quat_decompose_swing_twist(self, q: np.ndarray, axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ax = axis / (np.linalg.norm(axis) + 1e-12)
        qv = q[:3]
        twist_axis = ax * (np.dot(qv, ax))
        q_twist = np.array([twist_axis[0], twist_axis[1], twist_axis[2], q[3]], dtype=np.float64)
        q_twist /= (np.linalg.norm(q_twist) + 1e-12)
        q_twist_conj = np.array([-q_twist[0], -q_twist[1], -q_twist[2], q_twist[3]])
        swing = self._quat_mul(q, q_twist_conj)
        swing /= (np.linalg.norm(swing) + 1e-12)
        return swing, q_twist

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x1,y1,z1,w1 = a
        x2,y2,z2,w2 = b
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], dtype=np.float64)

    @staticmethod
    def _tri_area2_norm(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Half of absolute 2D cross product magnitude for triangle area in normalized coords."""
        ab = b - a; ac = c - a
        return 0.5 * abs(ab[0]*ac[1] - ab[1]*ac[0])

    def _check_led_match(
        self,
        model_led_indices: List[int],
        max_blob_depth: int,
        use_pose_prior: bool,
        pose_prior: Optional[Tuple[np.ndarray, np.ndarray]],
        pos_error_thresh: Optional[np.ndarray],
        rot_error_thresh: Optional[np.ndarray],
        use_gravity: bool,
        gravity_vec: Optional[np.ndarray],
        swing_prior: Optional[np.ndarray],
        gravity_tolerance_rad: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        num_trials = 0
        num_pose_checks = 0
        improved_any = False
        best_metrics = None
        best_R = None
        best_t = None
        best_blob_depth = -1

        led0, led1, led2, led3 = (self.leds[i] for i in model_led_indices)
        Xbuf = self._p3p_obj; Xbuf[0,:]=led0.pos; Xbuf[1,:]=led1.pos; Xbuf[2,:]=led2.pos
        d0, d1, d2 = led0.dir, led1.dir, led2.dir
        xcheck = led3.pos

        for b_anchor_idx in range(len(self._points)):
            anchor = self._points[b_anchor_idx]
            neigh = anchor.neighbours[:max_blob_depth]
            if len(neigh) < 3:
                continue
            L = len(neigh)
            for i in range(L):
                for j in range(i+1, L):
                    for k in range(j+1, L):
                        ip0 = self._points[b_anchor_idx].point_homog[:2]
                        ip1 = self._points[neigh[i]].point_homog[:2]
                        ip2 = self._points[neigh[j]].point_homog[:2]
                        ip3 = self._points[neigh[k]].point_homog[:2]

                        area = self._tri_area2_norm(ip0, ip1, ip2)
                        if area < MIN_TRI_AREA_NORM:
                            continue

                        def _px(p):
                            return np.array([p[0]*self.fx + self.cx, p[1]*self.fy + self.cy], dtype=np.float64)
                        p0 = _px(ip0); p1 = _px(ip1); p2 = _px(ip2)
                        if (np.linalg.norm(p0 - p1) < MIN_PAIR_SEP_PX or
                            np.linalg.norm(p0 - p2) < MIN_PAIR_SEP_PX or
                            np.linalg.norm(p1 - p2) < MIN_PAIR_SEP_PX):
                            continue

                        Ibuf = self._p3p_img; Ibuf[0,:]=ip0; Ibuf[1,:]=ip1; Ibuf[2,:]=ip2
                        num_trials += 1
                        retval, rvecs, tvecs = cv2.solveP3P(
                            Xbuf, Ibuf, self._eye3, None,
                            flags=cv2.SOLVEPNP_LAMBDA_TWIST if hasattr(cv2, 'SOLVEPNP_LAMBDA_TWIST') else cv2.SOLVEPNP_P3P,
                        )
                        if not retval:
                            continue
                        for idx in range(len(rvecs)):
                            rvec = rvecs[idx].reshape(3)
                            tvec = tvecs[idx].reshape(3)
                            R, _ = cv2.Rodrigues(rvec)
                            if not (self.z_min <= tvec[2] <= self.z_max):
                                continue
                            if use_gravity:
                                q = quat_from_R(R)
                                swing, _tw = self._quat_decompose_swing_twist(q, gravity_vec)
                                ang_pose = 2.0 * math.acos(max(-1.0, min(1.0, swing[3])))
                                ang_prior = 2.0 * math.acos(max(-1.0, min(1.0, swing_prior[3])))
                                if abs(ang_pose - ang_prior) > gravity_tolerance_rad:
                                    continue
                            ok3 = True
                            for (Xw, dirw, ip) in ((led0.pos, d0, ip0), (led1.pos, d1, ip1), (led2.pos, d2, ip2)):
                                cam_p = (R @ Xw) + tvec
                                if cam_p[2] <= 0.0:
                                    ok3 = False; break
                                view_dir = cam_p / (np.linalg.norm(cam_p) + 1e-12)
                                cam_dir = R @ dirw[:3]; cam_dir /= (np.linalg.norm(cam_dir) + 1e-12)
                                if float(np.dot(view_dir, cam_dir)) > 0.0:
                                    ok3 = False; break
                                proj = cam_p / cam_p[2]
                                if float(np.linalg.norm(proj[:2] - ip)) > self.backcheck_tol_norm:
                                    ok3 = False; break
                            if not ok3:
                                continue
                            cam_p4 = (R @ xcheck) + tvec
                            if cam_p4[2] <= 0.0:
                                continue
                            proj4 = cam_p4 / cam_p4[2]
                            if float(np.linalg.norm(proj4[:2] - ip3)) > self._points[neigh[k]].max_dist:
                                continue

                            #collect greedy 1-1 gated matches (pixel domain) from this seed
                            P = R @ self._led_pos_T + tvec.reshape(3,1)
                            zpos = P[2,:] > 0.0
                            dirs_cam = R @ self._led_dir_T
                            view_dirs = P / (np.linalg.norm(P, axis=0, keepdims=True) + 1e-12)
                            facing = (np.sum(dirs_cam * view_dirs, axis=0) < 0.05)  #allow slight grazing
                            vis = zpos & facing
                            if not np.any(vis):
                                continue

                            Pv = P[:, vis]
                            proj_norm = Pv[:2,:] / Pv[2:3,:]
                            proj_px = (self.K[:2,:2] @ proj_norm) + self.K[:2,2:3]
                            proj_px = proj_px.T  #(Nv,2)

                            blob_xy = np.ascontiguousarray([[b.x, b.y] for b in self._blobs], dtype=np.float64)
                            scale = self.gate_px_scale * (self.fx + self.fy)

                            obj_pts = []
                            img_pts = []
                            used_blob = np.zeros(len(self._blobs), dtype=np.bool_)
                            vis_idx = np.nonzero(vis)[0]
                            for ii, led_idx in enumerate(vis_idx):
                                p = proj_px[ii]
                                dxy = blob_xy - p
                                d2 = np.einsum('ij,ij->i', dxy, dxy)
                                bidx = int(np.argmin(d2))
                                if used_blob[bidx]:
                                    continue
                                gate_px = float(self._points[bidx].max_dist) * scale
                                if math.sqrt(float(d2[bidx])) <= gate_px:
                                    used_blob[bidx] = True
                                    obj_pts.append(self.leds[led_idx].pos.astype(np.float64))
                                    img_pts.append(p.astype(np.float64))

                            if len(obj_pts) >= 6:
                                obj = np.asarray(obj_pts, np.float64).reshape(-1,1,3)
                                img = np.asarray(img_pts, np.float64).reshape(-1,1,2)
                                try:
                                    rvec_ref, _ = cv2.Rodrigues(R)
                                    rvec_ref = rvec_ref.reshape(3,1)
                                    tvec_ref = tvec.reshape(3,1)
                                    rvec_ref, tvec_ref = cv2.solvePnPRefineLM(obj, img, self.K, self.dist, rvec_ref, tvec_ref)
                                    R, _ = cv2.Rodrigues(rvec_ref)
                                    tvec = tvec_ref.reshape(3)
                                except Exception:
                                    #if refine is unavailable, just keep the seed
                                    pass

                            #full evaluation and bookkeeping
                            num_pose_checks += 1
                            metrics = self._evaluate_pose(R, tvec, use_pose_prior, pose_prior, pos_error_thresh, rot_error_thresh)
                            if self._is_better_pose(best_metrics, metrics):
                                best_metrics = metrics; best_R = R; best_t = tvec; best_blob_depth = 1
                                improved_any = True
        return improved_any, {
            'metrics': best_metrics,
            'R': best_R,
            't': best_t,
            'num_trials': num_trials,
            'num_pose_checks': num_pose_checks,
            'blob_depth': best_blob_depth,
            'led_depth': None,
        }

    def _evaluate_pose(
        self,
        R: np.ndarray,
        t: np.ndarray,
        use_pose_prior: bool,
        pose_prior: Optional[Tuple[np.ndarray, np.ndarray]],
        pos_err_thresh: Optional[np.ndarray],
        rot_err_thresh: Optional[np.ndarray],
    ) -> PoseMetrics:
        if use_pose_prior and pose_prior is not None and (pos_err_thresh is not None or rot_err_thresh is not None):
            R0, t0 = self._parse_pose_prior(pose_prior)
            if pos_err_thresh is not None:
                if np.any(np.abs(t - t0) > pos_err_thresh.reshape(-1)):
                    return PoseMetrics()
            if rot_err_thresh is not None:
                dR = R0.T @ R
                angle = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1.0) * 0.5)))
                if angle > float(rot_err_thresh.reshape(-1)[0]):
                    return PoseMetrics()
        #preallocate projection buffer
        N = self.N_leds
        if self._proj_buf is None or self._proj_buf.shape != (3, N):
            self._proj_buf = np.empty((3, N), np.float64)
        P = self._proj_buf
        # P = R*X + t
        P[...] = R @ self._led_pos_T
        P += t.reshape(3, 1)
        vis = P[2, :] > 0.0
        if not np.any(vis):
            return PoseMetrics()
        dirs_cam = R @ self._led_dir_T
        view_dirs = P / (np.linalg.norm(P, axis=0, keepdims=True) + 1e-12)
        facing = (np.sum(dirs_cam * view_dirs, axis=0) < 0.05) 
        vis = vis & facing
        if not np.any(vis):
            return PoseMetrics()
        Pv = P[:, vis]
        xnorm = Pv[:2, :] / Pv[2:3, :]
        xp = (self.K[:2, :2] @ xnorm) + self.K[:2, 2:3]
        xp = xp.T  # (Nvis,2)
        blob_xy = np.ascontiguousarray([[b.x, b.y] for b in self._blobs], dtype=np.float64)

        #global cheapest-first 1-to-1 matching within per-blob gates
        pairs: List[Tuple[float,int,int]] = []
        scale = self.gate_px_scale * (self.fx + self.fy)
        for i in range(xp.shape[0]):
            dxy = blob_xy - xp[i]
            d2  = np.einsum('ij,ij->i', dxy, dxy)
            for bi, dd in enumerate(d2):
                gate_px = float(self._points[bi].max_dist) * scale
                if dd <= (gate_px * gate_px):  #squared compare
                    pairs.append((float(dd), i, bi))

        pairs.sort(key=lambda t_: t_[0])
        used_blob = np.zeros(len(self._blobs), dtype=np.bool_)
        used_led  = np.zeros(xp.shape[0], dtype=np.bool_)
        err_sum = 0.0; matched = 0
        for dd, li, bi in pairs:
            if used_led[li] or used_blob[bi]:
                continue
            used_led[li] = True
            used_blob[bi] = True
            err_sum += dd
            matched += 1

        metrics = PoseMetrics()
        metrics.visible_leds = int(xp.shape[0])
        metrics.matched_blobs = int(matched)
        metrics.unmatched_blobs = int(len(self._blobs) - matched)
        metrics.reprojection_error = float(err_sum)
        if matched >= 4:
            metrics.match_flags |= POSE_MATCH_GOOD
        if matched >= self.strong_min_inliers:
            rmse = math.sqrt(err_sum / max(matched, 1))
            if rmse <= self.strong_max_rmse_px:
                metrics.match_flags |= POSE_MATCH_STRONG
        return metrics

    @staticmethod
    def _is_better_pose(best: Optional[PoseMetrics], cand: Optional[PoseMetrics]) -> bool:
        if cand is None:
            return False
        if best is None:
            return True
        if cand.matched_blobs != best.matched_blobs:
            return cand.matched_blobs > best.matched_blobs
        if cand.reprojection_error != best.reprojection_error:
            return cand.reprojection_error < best.reprojection_error
        return cand.visible_leds > best.visible_leds

    def warmup(self):
        """Run a quick, no-op path to JIT-compile numba functions early."""
        if NUMBA_AVAILABLE:
            _ = _sorted_sqdist_indices(np.zeros((3,2), np.float64))


def make_search(K: np.ndarray, dist: np.ndarray, model_points: np.ndarray, model_normals: np.ndarray, model_id: int = 0) -> CorrespondenceSearch:
    return CorrespondenceSearch(K, dist, model_points, model_normals, model_id=model_id)

if __name__ == '__main__':
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros(5)
    pts = np.array([[-0.05,-0.05,0],[0.05,-0.05,0],[0.05,0.05,0],[-0.05,0.05,0],[0.0,0.0,0]], dtype=np.float64)
    nrms = np.tile(np.array([0,0,-1.0], dtype=np.float64), (pts.shape[0],1))
    cs = CorrespondenceSearch(K, dist, pts, nrms)
    #project a synthetic pose
    R, _ = cv2.Rodrigues(np.array([0.05,-0.02,0.01]))
    t = np.array([0.0,0.0,1.0])
    P = (R @ pts.T + t.reshape(3,1))
    xnorm = P[:2,:] / P[2:3,:]
    pix = (K[:2,:2] @ xnorm) + K[:2,2:3]
    pix = pix.T
    blobs = [Blob(x=float(pix[i,0]), y=float(pix[i,1]), width=6.0, height=6.0) for i in range(pix.shape[0])]
    cs.set_blobs(blobs, search_flags=CS_FLAG_MATCH_ALL_BLOBS)
    ok, res = cs.solve()
    print('OK:', ok, 'inliers:', res['metrics'].matched_blobs if ok else 0)
