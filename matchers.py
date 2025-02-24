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
                matches.append((r, c))
                unmatched_old.discard(r)
                unmatched_new.discard(c)
                
    return matches, unmatched_old, unmatched_new