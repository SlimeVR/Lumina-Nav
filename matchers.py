import numpy as np 
from scipy.optimize import linear_sum_assignment

#TODO change this to use KLT like basalt rather than this 
def match_blobs_sumassign(old_blobs, new_blobs, max_match_distance=50.0):
    N = len(old_blobs)
    M = len(new_blobs)

    cost_matrix = np.zeros((N, M), dtype=np.float32)
    for i, old_b in enumerate(old_blobs):
        for j, new_b in enumerate(new_blobs):
            dx = old_b.x - new_b.x
            dy = old_b.y - new_b.y
            dist = np.sqrt(dx * dx + dy * dy)
            cost_matrix[i, j] = dist
    if N > M:
        pad_width = N - M
        cost_matrix = np.hstack([cost_matrix, np.full((N, pad_width), fill_value=1e9)])
    elif M > N:
        pad_height = M - N
        cost_matrix = np.vstack([cost_matrix, np.full((pad_height, M), fill_value=1e9)])

    row_ind, col_ind = linear_sum_assignment(cost_matrix) # i think this is an imp of the Hungarian Algorithm?
    matches = []
    unmatched_old = set(range(N))
    unmatched_new = set(range(M))

    for r, c in zip(row_ind, col_ind):
        if r < N and c < M:
            dist = cost_matrix[r, c]
            if dist < max_match_distance:
                print(dist)
                matches.append((r, c))
                unmatched_old.discard(r)
                unmatched_new.discard(c)
                
    return matches, unmatched_old, unmatched_new


import numpy as np
import random

import numpy as np
import random

def match_blobs_ransac(old_blobs, new_blobs, max_match_distance=50.0, 
                        ransac_threshold=5.0, num_iterations=1000, 
                        size_threshold=0.2):
    """
    Match blobs using a RANSAC strategy, considering both position and size similarity.
    
    Parameters:
      old_blobs: list of objects with attributes x, y, and area.
      new_blobs: list of objects with attributes x, y, and area.
      max_match_distance: maximum allowed distance for candidate pairings.
      ransac_threshold: error threshold to consider a candidate an inlier with respect to the estimated translation.
      num_iterations: number of random samples to try in RANSAC.
      size_threshold: maximum allowed relative difference in area for matching blobs.
      
    Returns:
      matches: list of tuples (i, j) where i is index into old_blobs and j into new_blobs.
      unmatched_old: set of indices in old_blobs not matched.
      unmatched_new: set of indices in new_blobs not matched.
    """
    # Build candidate matches: every pairing within max_match_distance and similar size is a candidate.
    candidate_matches = []
    for i, old_b in enumerate(old_blobs):
        for j, new_b in enumerate(new_blobs):
            dx = new_b.x - old_b.x
            dy = new_b.y - old_b.y
            dist = np.sqrt(dx**2 + dy**2)
            size_diff = abs(new_b.area - old_b.area) / max(old_b.area, new_b.area)

            if dist < max_match_distance and size_diff < size_threshold:
                candidate_matches.append((i, j, dx, dy, size_diff))

    # If no candidate pairs are found, return all blobs as unmatched.
    if not candidate_matches:
        return [], set(range(len(old_blobs))), set(range(len(new_blobs)))
    
    best_inliers = []
    best_model = None  # Best translation (tx, ty)
    
    # RANSAC iterations: randomly select one candidate as the hypothesized translation.
    for _ in range(num_iterations):
        sample = random.choice(candidate_matches)
        tx, ty = sample[2], sample[3]
        current_inliers = []
        
        # Check each candidate match against the hypothesized translation and size.
        for cand in candidate_matches:
            dx, dy, size_diff = cand[2], cand[3], cand[4]
            # If the difference in translation is small and size is similar, count it as an inlier.
            if np.sqrt((dx - tx)**2 + (dy - ty)**2) < ransac_threshold and size_diff < size_threshold:
                current_inliers.append(cand)
        
        # Keep track of the model with the most inliers.
        if len(current_inliers) > len(best_inliers):
            best_inliers = current_inliers
            best_model = (tx, ty)
    
    # Build final matches from the best inliers, ensuring one-to-one matching.
    matches = []
    matched_old = set()
    matched_new = set()
    for cand in best_inliers:
        i, j = cand[0], cand[1]
        if i not in matched_old and j not in matched_new:
            matches.append((i, j))
            matched_old.add(i)
            matched_new.add(j)
    
    unmatched_old = set(range(len(old_blobs))) - matched_old
    unmatched_new = set(range(len(new_blobs))) - matched_new
    return matches, unmatched_old, unmatched_new
