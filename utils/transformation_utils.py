import numpy as np
def skew(x):
    X = np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])
    return X

def exp_so3(w_3x1):
    angle = np.linalg.norm(w_3x1)

    if np.abs(angle) < 1e-8:
        skew_w = skew(w_3x1)
        return np.identity(3) + skew_w
    
    axis = w_3x1 / angle
    skew_axis = skew(axis)
    s = np.sin(angle)
    c = np.cos(angle)
    return c * np.identity(3) + s * skew_axis + (1 - c) * np.outer(axis, axis)

def log_SO3(R):
    angle = np.arccos(max(-1.0, min(1.0, 0.5 * (np.trace(R) - 1.0))))
    if np.abs(angle) < 1e-8:
        skew_w = 0.5 * (R - R.transpose())
        return np.array([skew_w[2,1], skew_w[0,2], skew_w[1,0]])
    
    s = np.sin(angle)
    c = np.cos(angle)
    skew_w = (angle / (2.0 *s)) * (R - R.transpose())
    return np.array([skew_w[2,1], skew_w[0,2], skew_w[1,0]])

def SE3(T, R):
    RT = np.zeros((4, 4))
    RT[:3, :3] = R
    RT[:3, 3] = T.reshape(3)
    RT[3, 3] = 1
    return RT

def inv_SE3_T_R(T, R):
    return SE3(np.dot(-R.transpose(),T), R.transpose())

def inv_SE3_RT(RT):
    return inv_SE3_T_R(RT[:3, 3], RT[:3, :3])

def GetRelPose_tail2tail(RT01, RT02):
    R01 = RT01[:3, :3]
    T01 = RT01[:3, 3]
    R02 = RT02[:3, :3]
    T02 = RT02[:3, 3]
    R10 = R01.transpose()
    return SE3(np.dot(R10, (T02 - T01).reshape(3,1)), np.dot(R10, R02) )


