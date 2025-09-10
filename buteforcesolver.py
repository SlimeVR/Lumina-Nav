import itertools
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Set
from numba import jit, prange, njit
import time
@njit
def _fast_pairwise_dists_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast 2D pairwise distance computation"""
    Na = a.shape[0]
    Nb = b.shape[0]
    result = np.zeros((Na, Nb), dtype=np.float64)
    for i in range(Na):
        for j in range(Nb):
            dx = a[i, 0] - b[j, 0]
            dy = a[i, 1] - b[j, 1]
            result[i, j] = dx * dx + dy * dy
    return result

@njit
def _fast_pairwise_dists_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast 3D pairwise distance computation"""
    Na = a.shape[0]
    Nb = b.shape[0]
    result = np.zeros((Na, Nb), dtype=np.float64)
    for i in range(Na):
        for j in range(Nb):
            dx = a[i, 0] - b[j, 0]
            dy = a[i, 1] - b[j, 1]
            dz = a[i, 2] - b[j, 2]
            result[i, j] = dx * dx + dy * dy + dz * dz
    return result

@njit
def _fast_norm_squared(a, b):
    """Fast squared Euclidean distance between two 2D points"""
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

@njit
def _fast_score_correspondences(proj_points, centers_px, radii, facing_mask):
    N = proj_points.shape[0]
    M = centers_px.shape[0]
    
    used_blob = np.zeros(M, dtype=np.bool_)
    inliers = 0
    sqerr = 0.0
    
    #pre-allocate arrays for correspondences
    led_indices = np.full(N, -1, dtype=np.int32)
    det_indices = np.full(N, -1, dtype=np.int32)
    correspondence_count = 0
    
    for i in range(N):
        if not facing_mask[i]:
            continue
            
        #find nearest unused blob
        best_distance_sq = np.inf
        best_blob_idx = -1
        
        for j in range(M):
            if used_blob[j]:
                continue
            dist_sq = _fast_norm_squared(proj_points[i], centers_px[j])
            if dist_sq < best_distance_sq:
                best_distance_sq = dist_sq
                best_blob_idx = j
        
        if best_blob_idx >= 0:
            dist = np.sqrt(best_distance_sq)
            if dist <= radii[best_blob_idx]:
                used_blob[best_blob_idx] = True
                inliers += 1
                sqerr += best_distance_sq
                
                #store correspondence
                led_indices[correspondence_count] = i
                det_indices[correspondence_count] = best_blob_idx
                correspondence_count += 1
    
    return inliers, sqerr, led_indices[:correspondence_count], det_indices[:correspondence_count]

@njit
def _fast_compute_scale_ratios(X, led_corr_items, centers_px, distance_to_target, focal_length, max_pairs=10):
    scale_ratios = np.zeros(max_pairs, dtype=np.float64)
    ratio_count = 0
    
    n_corr = len(led_corr_items)
    for i in range(n_corr):
        if ratio_count >= max_pairs:
            break
        for j in range(i + 1, n_corr):
            if ratio_count >= max_pairs:
                break
                
            led1_idx = led_corr_items[i, 0]
            det1_idx = led_corr_items[i, 1]
            led2_idx = led_corr_items[j, 0]
            det2_idx = led_corr_items[j, 1]
            
            # 3D distance between LEDs
            dx = X[led1_idx, 0] - X[led2_idx, 0]
            dy = X[led1_idx, 1] - X[led2_idx, 1]
            dz = X[led1_idx, 2] - X[led2_idx, 2]
            world_dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if world_dist < 15.0:  #skip very close LED pairs
                continue
            
            #2D distance between detections
            px_dx = centers_px[det1_idx, 0] - centers_px[det2_idx, 0]
            px_dy = centers_px[det1_idx, 1] - centers_px[det2_idx, 1]
            pixel_dist = np.sqrt(px_dx*px_dx + px_dy*px_dy)
            
            #expected pixel distance
            expected_pixel_dist = (world_dist * focal_length) / distance_to_target
            
            if expected_pixel_dist > 1.0:
                scale_ratios[ratio_count] = pixel_dist / expected_pixel_dist
                ratio_count += 1
    
    return scale_ratios[:ratio_count]


class SolvePoseBruteOptimized:
    def __init__(
        self,
        K: np.ndarray,
        dist: Optional[np.ndarray],
        world_points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        max_led_depth: int = 8,
        max_blob_depth: int = 8,
        shallow_search: bool = True,
        stop_on_strong: bool = True,
        strong_min_inliers: int = 7,
        strong_px_err_per_inlier: float = 1.5,
        adaptive_search: bool = True,
        max_iterations: int = 10000,
        use_temporal: bool = True, 
        distance_adaptive: bool = True,
    ):
        self.K = K.astype(np.float64)
        self.dist = np.zeros((5, 1), np.float64) if dist is None else dist.astype(np.float64).reshape(-1, 1)
        self.X = world_points.astype(np.float64)
        self.normals = normals.astype(np.float64) if normals is not None else None

        self.N = self.X.shape[0]
        self.max_led_depth = int(max(3, max_led_depth))
        self.max_blob_depth = int(max(3, max_blob_depth))
        self.shallow_search = shallow_search
        self.stop_on_strong = stop_on_strong
        self.strong_min_inliers = strong_min_inliers
        self.strong_px_err_per_inlier = strong_px_err_per_inlier
        self.adaptive_search = adaptive_search
        self.max_iterations = max_iterations
        self.use_temporal = use_temporal
        self.distance_adaptive = distance_adaptive

        d3 = _fast_pairwise_dists_3d(self.X, self.X)
        order = np.argsort(d3, axis=1)
        keep = min(self.N, 1 + self.max_led_depth)
        self.led_neighbors: List[np.ndarray] = [order[i, :keep] for i in range(self.N)]
        
        self._projection_cache = {}
        self._last_pose_hash = None
        
        self._last_rvec = None
        self._last_tvec = None
        self._last_distance = None
        self._last_inlier_count = None
        
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.avg_focal_length = (self.fx + self.fy) / 2.0
        
        if self.N > 1:
            led_distances = []
            for i in range(min(10, self.N)):
                for j in range(i+1, min(10, self.N)):
                    led_distances.append(np.linalg.norm(self.X[i] - self.X[j]))
            self.avg_led_spacing = np.median(led_distances) if led_distances else 30.0
        else:
            self.avg_led_spacing = 30.0
        
        self.stats = {
            'iterations_tested': 0,
            'p3p_calls': 0,
            'early_terminations': 0,
            'cache_hits': 0,
            'temporal_success': 0,
            'solve_time': 0.0
        }

    def _get_pose_hash(self, rvec, tvec):
        return hash((tuple(rvec.flatten()), tuple(tvec.flatten())))

    @staticmethod
    def _centers_from_detections(dets: List[Dict], assume_top_left: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        M = len(dets)
        centers = np.zeros((M, 2), dtype=np.float64)
        sizes = np.zeros((M,), dtype=np.float64)
        for i, d in enumerate(dets):
            x, y = float(d["x"]), float(d["y"])
            inner = float(d["inner"])
            if assume_top_left:
                cx = x + inner * 0.5
                cy = y + inner * 0.5
            else:
                cx, cy = x, y
            centers[i] = (cx, cy)
            sizes[i] = inner
        return centers, sizes

    def _make_blob_neighbors(self, centers: np.ndarray) -> List[np.ndarray]:
        M = centers.shape[0]
        if M == 0:
            return []
        d2 = _fast_pairwise_dists_2d(centers, centers)
        order = np.argsort(d2, axis=1)
        keep = min(M, 1 + self.max_blob_depth)
        return [order[i, :keep] for i in range(M)]

    def _estimate_adaptive_radius(self, inner_sizes: np.ndarray, expected_distance: Optional[float] = None) -> np.ndarray:
        if self.distance_adaptive and expected_distance is not None:
            #estimate pixel size of LEDs at this distance
            expected_pixel_size = (self.avg_led_spacing * self.fx) / expected_distance
            
            #set tolerance based on expected size with margin
            adaptive_radius = min(50.0, max(10.0, expected_pixel_size * 2.5))
            radii = np.full(len(inner_sizes), adaptive_radius, dtype=np.float64)
        else:
            radii = np.maximum(inner_sizes * 2.0, 15.0)
            radii = np.minimum(radii, 50.0)  #cap at 50 pixels
        
        return radii

    def _try_temporal_refinement(
        self, 
        centers_px: np.ndarray, 
        radii: np.ndarray,
        max_movement: float = 100.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        if not self.use_temporal or self._last_rvec is None or self._last_tvec is None:
            return None
            
        try:
            proj, _ = cv2.projectPoints(self.X, self._last_rvec, self._last_tvec, self.K, self.dist)
            proj_points = proj.reshape(-1, 2).astype(np.float64)
        except cv2.error:
            return None
        
        #check facing
        if self.normals is not None:
            R, _ = cv2.Rodrigues(self._last_rvec)
            Xc = (R @ self.X.T + self._last_tvec).T
            Nc = (R @ self.normals.T).T
            view_dir = Xc / np.maximum(1e-9, np.linalg.norm(Xc, axis=1, keepdims=True))
            facing = (np.sum(view_dir * Nc, axis=1) <= 0.0)
        else:
            facing = np.ones(self.N, dtype=np.bool_)
        
        inliers, sqerr, led_indices, det_indices = _fast_score_correspondences(
            proj_points, centers_px, radii * 0.7, facing
        )
        
        if inliers >= max(4, self._last_inlier_count * 0.7 if self._last_inlier_count else 4):
            correspondences = {}
            object_points = []
            image_points = []
            
            for i in range(len(led_indices)):
                led_idx = int(led_indices[i])
                det_idx = int(det_indices[i])
                correspondences[led_idx] = det_idx
                object_points.append(self.X[led_idx])
                image_points.append(centers_px[det_idx])
            
            if len(object_points) >= 4:
                object_points = np.array(object_points, dtype=np.float64)
                image_points = np.array(image_points, dtype=np.float64).reshape(-1, 1, 2)
                
                try:
                    success, refined_rvec, refined_tvec = cv2.solvePnP(
                        object_points, image_points, self.K, self.dist,
                        rvec=self._last_rvec.copy(), tvec=self._last_tvec.copy(),
                        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if success:
                        movement = np.linalg.norm(refined_tvec - self._last_tvec)
                        if movement < max_movement:
                            self.stats['temporal_success'] += 1
                            return refined_rvec, refined_tvec, correspondences
                except cv2.error:
                    pass
        
        return None

    def _validate_pose_geometry(
        self, 
        rvec: np.ndarray, 
        tvec: np.ndarray, 
        correspondences: Dict[int, int], 
        centers_px: np.ndarray
    ) -> bool:
        if len(correspondences) < 3:
            return False
            
        distance_to_target = np.linalg.norm(tvec)
        
        if distance_to_target < 50.0 or distance_to_target > 5000.0:
            return False
        
        corr_array = np.array(list(correspondences.items()), dtype=np.int32)
        
        scale_ratios = _fast_compute_scale_ratios(
            self.X, corr_array, centers_px, 
            distance_to_target, self.avg_focal_length, max_pairs=10
        )
        
        if len(scale_ratios) == 0:
            return True
        
        median_ratio = np.median(scale_ratios)
        std_ratio = np.std(scale_ratios)
        
        distance_factor = min(2.0, distance_to_target / 500.0)
        
        min_scale = 0.3 / max(1.0, distance_factor)
        max_scale = 3.0 * max(1.0, distance_factor)
        
        if median_ratio < min_scale or median_ratio > max_scale:
            return False
        
        max_allowed_std = 0.3 + (0.2 * distance_factor)
        if std_ratio > max_allowed_std:
            return False
        
        outlier_count = np.sum(np.abs(scale_ratios - median_ratio) > 2.0 * max(std_ratio, 0.1))
        if outlier_count > len(scale_ratios) * 0.4:
            return False
            
        return True

    def _score_pose_fast(
        self, 
        rvec: np.ndarray, 
        tvec: np.ndarray, 
        centers_px: np.ndarray, 
        radii: np.ndarray
    ) -> Tuple[int, float, Dict]:

        pose_hash = self._get_pose_hash(rvec, tvec)
        if pose_hash == self._last_pose_hash and pose_hash in self._projection_cache:
            proj_points = self._projection_cache[pose_hash]
            self.stats['cache_hits'] += 1
        else:
            try:
                proj, _ = cv2.projectPoints(self.X, rvec, tvec, self.K, self.dist)
                proj_points = proj.reshape(-1, 2).astype(np.float64)

                if len(self._projection_cache) > 100:
                    self._projection_cache.clear()
                self._projection_cache[pose_hash] = proj_points
                self._last_pose_hash = pose_hash
            except cv2.error:
                return 0, np.inf, {}

        if self.normals is not None:
            R, _ = cv2.Rodrigues(rvec)
            Xc = (R @ self.X.T + tvec).T
            Nc = (R @ self.normals.T).T
            view_dir = Xc / np.maximum(1e-9, np.linalg.norm(Xc, axis=1, keepdims=True))
            facing = (np.sum(view_dir * Nc, axis=1) <= 0.0)
        else:
            facing = np.ones(self.N, dtype=np.bool_)

        inliers, sqerr, led_indices, det_indices = _fast_score_correspondences(
            proj_points, centers_px, radii, facing
        )
        
        rmse = float(np.sqrt(sqerr / max(1, inliers)))
    
        correspondences = {}
        for i in range(len(led_indices)):
            correspondences[int(led_indices[i])] = int(det_indices[i])
        
        return inliers, rmse, correspondences

    @staticmethod
    def _is_better(inliers: int, rmse: float, best_inliers: int, best_rmse: float) -> bool:
        if inliers > best_inliers:
            return True
        if inliers == best_inliers and rmse < best_rmse:
            return True
        return False

    def _search_with_depths(
        self, 
        max_led_depth: int,
        max_blob_depth: int,
        centers_px: np.ndarray,
        radii: np.ndarray,
        blob_neighbors: List[np.ndarray],
        best: Dict,
        distance_hint: Optional[float] = None
    ) -> bool:
        iterations = 0
        found_improvement = False
        
        #adaptive limits based on distance
        if self.distance_adaptive and distance_hint is not None:
            if distance_hint > 1500:  #far
                blob_anchor_limit = min(3, len(blob_neighbors))
                led_combination_limit = 10
                blob_combination_limit = 5
            elif distance_hint > 800:  #med
                blob_anchor_limit = min(4, len(blob_neighbors))
                led_combination_limit = 15
                blob_combination_limit = 8
            else:  #close
                blob_anchor_limit = min(5, len(blob_neighbors))
                led_combination_limit = 20
                blob_combination_limit = 10
        else:
            blob_anchor_limit = min(5, len(blob_neighbors))
            led_combination_limit = 20
            blob_combination_limit = 10
        
        #sort LED anchors by connectivity
        led_anchor_order = list(range(len(self.led_neighbors)))
        led_anchor_order.sort(key=lambda i: len(self.led_neighbors[i]), reverse=True)
        
        #limit LED anchors
        max_led_anchors = min(10, len(led_anchor_order))
        
        for li in led_anchor_order[:max_led_anchors]:
            led_nbrs_full = self.led_neighbors[li]
            if len(led_nbrs_full) < 4:
                continue
                
            led_nbrs = led_nbrs_full[1: 1 + max_led_depth]
            if led_nbrs.shape[0] < 3:
                continue

            #early termination check
            if iterations > self.max_iterations:
                self.stats['early_terminations'] += 1
                break

            #generate LED combinations
            led_combinations = list(itertools.combinations(led_nbrs, 3))
            if len(led_combinations) > led_combination_limit:
                #sample combinations uniformly
                indices = np.linspace(0, len(led_combinations)-1, led_combination_limit, dtype=int)
                led_combinations = [led_combinations[i] for i in indices]

            for (j1, j2, j3) in led_combinations:
                led_quad = np.array([li, j1, j2, j3], dtype=int)

                for perm in [(0, 1, 2, 3), (0, 2, 1, 3)]:
                    led_idx = led_quad[list(perm)]
                    X3 = self.X[led_idx[:3]]
                    X4 = self.X[led_idx[3]]

                    for bi in range(blob_anchor_limit):
                        blob_nbrs_full = blob_neighbors[bi]
                        if len(blob_nbrs_full) < 4:
                            continue
                            
                        blob_nbrs = blob_nbrs_full[1: 1 + max_blob_depth]
                        if blob_nbrs.shape[0] < 3:
                            continue

                        #generate blob combinations
                        blob_combinations = list(itertools.combinations(blob_nbrs, 3))
                        if len(blob_combinations) > blob_combination_limit:
                            indices = np.linspace(0, len(blob_combinations)-1, blob_combination_limit, dtype=int)
                            blob_combinations = [blob_combinations[i] for i in indices]

                        for (k1, k2, k3) in blob_combinations:
                            iterations += 1
                            if iterations > self.max_iterations:
                                break
                                
                            blob_quad = np.array([bi, k1, k2, k3], dtype=int)

                            #P3P solve
                            img3 = centers_px[blob_quad[:3]].reshape(-1, 1, 2)
                            self.stats['p3p_calls'] += 1
                            
                            try:
                                ok, rvecs, tvecs = cv2.solveP3P(
                                    X3.astype(np.float64), img3.astype(np.float64),
                                    self.K, self.dist, flags=cv2.SOLVEPNP_P3P
                                )
                            except cv2.error:
                                continue
                                
                            if not ok:
                                continue

                            #4th point validation
                            img4 = centers_px[blob_quad[3]].reshape(1, 1, 2)
                            r_tol = radii[blob_quad[3]]

                            for rv, tv in zip(rvecs, tvecs):
                                #Quick 4th point check
                                try:
                                    proj4, _ = cv2.projectPoints(X4.reshape(1, 3), rv, tv, self.K, self.dist)
                                    p4 = proj4.reshape(2)
                                    err4 = np.linalg.norm(p4 - img4.reshape(2))
                                    
                                    if err4 > r_tol:
                                        continue
                                except cv2.error:
                                    continue

                                #Full pose scoring
                                inliers, rmse, correspondences = self._score_pose_fast(rv, tv, centers_px, radii)
                                
                                #Apply distance prior if available
                                if self.distance_adaptive and distance_hint is not None and inliers > 0:
                                    actual_distance = np.linalg.norm(tv)
                                    distance_error = abs(actual_distance - distance_hint) / distance_hint
                                    if distance_error > 0.5:  # More than 50% off
                                        rmse *= (1.0 + distance_error * 0.5)  # Penalize
                                
                                if self._is_better(inliers, rmse, best["inliers"], best["reproj_rmse"]):
                                    # Additional geometry validation for significant improvements
                                    if inliers > best["inliers"] + 2 or (inliers == best["inliers"] and rmse < best["reproj_rmse"] * 0.7):
                                        if self._validate_pose_geometry(rv, tv, correspondences, centers_px):
                                            best.update({
                                                "inliers": inliers,
                                                "reproj_rmse": rmse,
                                                "rvec": rv.copy(),
                                                "tvec": tv.copy(),
                                                "led_correspondences": correspondences,
                                            })
                                            found_improvement = True
                                    else:
                                        #Minor improvement, accept without validation
                                        best.update({
                                            "inliers": inliers,
                                            "reproj_rmse": rmse,
                                            "rvec": rv.copy(),
                                            "tvec": tv.copy(),
                                            "led_correspondences": correspondences,
                                        })
                                        found_improvement = True

                                    #Dynamic early termination based on distance
                                    if self.stop_on_strong:
                                        if distance_hint is not None and distance_hint > 1000:
                                            # Lower requirements at distance
                                            if inliers >= max(5, self.strong_min_inliers - 2) and rmse <= self.strong_px_err_per_inlier * 1.5:
                                                self.stats['iterations_tested'] = iterations
                                                return True
                                        else:
                                            # Normal requirements
                                            if inliers >= self.strong_min_inliers and rmse <= self.strong_px_err_per_inlier:
                                                self.stats['iterations_tested'] = iterations
                                                return True

        self.stats['iterations_tested'] = iterations
        return found_improvement

    def solve(
        self,
        detections: List[Dict],
        frame_shape: Tuple[int, int],
        assume_top_left: bool = True,
        debug_draw_frame: Optional[np.ndarray] = None,
        expected_distance: Optional[float] = None,  # Hint for expected distance
        max_temporal_movement: float = 100.0,  # Max movement between frames
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Dict]:
        start_time = time.time()
        
        #reset stats
        self.stats = {k: 0 if k != 'solve_time' else 0.0 for k in self.stats}
        
        H, W = frame_shape[:2]
        if len(detections) < 4 or self.N < 4:
            return False, None, None, {"error": "insufficient_points"}

        #extract centers and setup
        centers_px, inner_sizes = self._centers_from_detections(detections, assume_top_left)
        
        #use last distance as hint if not provided
        if expected_distance is None and self._last_distance is not None:
            expected_distance = self._last_distance
        
        #calculate adaptive radii
        radii = self._estimate_adaptive_radius(inner_sizes, expected_distance)
        
        #try temporal refinement first
        temporal_result = self._try_temporal_refinement(centers_px, radii, max_temporal_movement)
        if temporal_result is not None:
            refined_rvec, refined_tvec, correspondences = temporal_result
            
            #update tracking
            self._last_rvec = refined_rvec
            self._last_tvec = refined_tvec
            self._last_distance = np.linalg.norm(refined_tvec)
            self._last_inlier_count = len(correspondences)
            
            self.stats['solve_time'] = time.time() - start_time
             
            return True, refined_rvec, refined_tvec, {
                "inliers": len(correspondences),
                "reproj_rmse": 0.0,  #could compute if needed
                "led_correspondences": correspondences,
                "used_temporal": True,
                **self.stats
            }
        
        #full search
        blob_neighbors = self._make_blob_neighbors(centers_px)
        if not blob_neighbors:
            return False, None, None, {"error": "no_blob_neighbors"}

        #best solution tracking
        best = {
            "inliers": -1,
            "reproj_rmse": np.inf,
            "rvec": None,
            "tvec": None,
            "led_correspondences": {},
        }

        #adaptive search depths based on distance
        if self.adaptive_search:
            if expected_distance is not None:
                if expected_distance > 1500:  #far
                    depth_schedules = [(2, 2), (3, 3)]
                elif expected_distance > 800:  #medium
                    depth_schedules = [(3, 3), (4, 4)]
                else:  #close
                    depth_schedules = [(3, 3), (4, 4), (5, 5)]
            else:
                #default adaptive schedule
                depth_schedules = [(2, 2), (3, 3), (4, 4)] if self.shallow_search else [(3, 3), (4, 5), (6, 6)]
        else:
            #fixed depths
            max_led_depth = 4 if self.shallow_search else self.max_led_depth
            max_blob_depth = 4 if self.shallow_search else self.max_blob_depth
            depth_schedules = [(max_led_depth, max_blob_depth)]

        #search with increasing complexity
        for led_depth, blob_depth in depth_schedules:
            found_good = self._search_with_depths(
                led_depth, blob_depth, centers_px, radii, 
                blob_neighbors, best, expected_distance
            )
            
            if found_good and self.stop_on_strong:
                #check if we should stop based on distance
                if expected_distance is not None and expected_distance > 1000:
                    if best["inliers"] >= 5 and best["reproj_rmse"] < 3.0:
                        break
                else:
                    if best["inliers"] >= self.strong_min_inliers and best["reproj_rmse"] <= self.strong_px_err_per_inlier:
                        break
            
            #general early termination
            if best["inliers"] >= 6 and best["reproj_rmse"] < 2.0:
                break

        #update temporal tracking
        if best["rvec"] is not None and best["tvec"] is not None:
            self._last_rvec = best["rvec"].copy()
            self._last_tvec = best["tvec"].copy()
            self._last_distance = np.linalg.norm(best["tvec"])
            self._last_inlier_count = best["inliers"]
        
        self.stats['solve_time'] = time.time() - start_time
        
        #prepare result
        ok = best["inliers"] >= 4 and np.isfinite(best["reproj_rmse"])
        
        result_score = best.copy()
        result_score.update(self.stats)
        if expected_distance is not None:
            result_score["expected_distance"] = expected_distance
            if best["tvec"] is not None:
                result_score["actual_distance"] = np.linalg.norm(best["tvec"])
        
        return ok, best["rvec"], best["tvec"], result_score