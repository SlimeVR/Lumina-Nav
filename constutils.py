from scipy import linalg
import numpy as np

def triangulate_dlt(P0, P1, x0, x1):

    """
    Triangulate one 2D-2D correspondence using the DLT method.
    :param P0: 3x4 projection matrix for camera0
    :param P1: 3x4 projection matrix for camera1
    :param x0: (u0, v0) in the left image (pixel coords)
    :param x1: (u1, v1) in the right image
    :return: (X, Y, Z) in 3D (not homogeneous)
    """

    u0, v0 = x0
    u1, v1 = x1
    p1 = P0[0, :]
    p2 = P0[1, :]
    p3 = P0[2, :]
    q1 = P1[0, :]
    q2 = P1[1, :]
    q3 = P1[2, :]
    A = np.vstack([
        u0 * p3 - p1,
        v0 * p3 - p2,
        u1 * q3 - q1,
        v1 * q3 - q2
    ])
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]