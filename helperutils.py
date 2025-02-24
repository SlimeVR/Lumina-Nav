import numpy as np

def load_intrinsics(filename):

    with open(filename, 'r') as f:
        lines = f.read().strip().splitlines()
    intrinsic_data = []
    distortion_data = []
    mode = None

    for line in lines:
        line = line.strip()
        if line.startswith('intrinsic'):
            mode = 'intrinsic'
            continue
        elif line.startswith('distortion'):
            mode = 'distortion'
            continue
        elif not line:
            continue
        if mode == 'intrinsic':
            row_vals = [float(x) for x in line.split()]
            intrinsic_data.append(row_vals)
        elif mode == 'distortion':
            distortion_data = [float(x) for x in line.split()]

    K = np.array(intrinsic_data, dtype=np.float32)
    dist = np.array(distortion_data, dtype=np.float32).reshape((1, -1))
    return K, dist




def load_extrinsics(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().splitlines()
    R_data = []
    T_data = []
    mode = None

    for line in lines:
        line = line.strip()
        if line.startswith('R:'):
            mode = 'R'
            continue
        elif line.startswith('T:'):
            mode = 'T'
            continue
        elif not line:
            continue
        if mode == 'R':
            row_vals = [float(x) for x in line.split()]
            R_data.append(row_vals)
        elif mode == 'T':
            row_vals = [float(x) for x in line.split()]
            T_data.append(row_vals)
            
    R = np.array(R_data, dtype=np.float32)
    T = np.array(T_data, dtype=np.float32)
    if T.shape == (3,):
        T = T.reshape(3, 1)
    return R, T
