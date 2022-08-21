import numpy as np
from numpy import cos,sin,pi,arccos

# def quaternion_rotate(v, axis, theta):
def quaternion_rotate(v, qu):
    """
    Args:
        v: 回転させるベクトル np.ndarray(3,1)
        qu: quaternion
    Returns:
        回転後のベクトル np.ndarray(3,1)
    """
    if qu[0] == 1: # u = (0,0,0)
        return v
    theta = 2*arccos(qu[0])
    axis = qu[1:]
    axis_norm = np.linalg.norm(axis)
    u = axis / axis_norm
    uv = np.dot(u, v)
    uxv = np.cross(u, v)
    return v*cos(theta) + (1 - cos(theta))*uv*u + sin(theta)*uxv

def to_quaternion(axis, angle):
    qu = np.zeros(4)
    qu[0] = cos(angle/2)
    u = axis/np.linalg.norm(axis)
    qu[1:] = u*np.sin(angle/2)
    return qu

def rotation_matrix(roll, pitch, yaw):
    """
    X-Y-Z系のオイラー角の回転行列
    Args:
        roll: ロール（x軸まわりの角度）
        pitch:ピッチ（y軸まわりの角度）
        yaw:ヨー（z軸まわりの角度）
    Returns:
        回転行列 np.ndarray(3,3)
    """
    Rx = np.array([[1,         0,         0],
                   [0, cos(roll),-sin(roll)],
                   [0, sin(roll), cos(roll)]])

    Ry = np.array([[ cos(pitch), 0, sin(pitch)],
                   [          0, 1,          0],
                   [-sin(pitch), 0, cos(pitch)]])

    Rz = np.array([[cos(yaw),-sin(yaw), 0],
                   [sin(yaw), cos(yaw), 0],
                   [       0,        0, 1]])

    R = np.matmul(Rz, Ry)
    R = np.matmul(R, Rx)
    return R

def rotate(R, p):
    """
    Args:
        R: 回転行列 np.ndarray(3,3)
        p: 回転させる点(np.ndarray(3,1))の集合 np.ndarray(3,N) ※Nは任意
    Returns:
        回転後の点の集合 np.ndarray(3,N)
    """
    return np.matmul(R, p)
