"""
点群のレジストレーションの初歩
https://qiita.com/shirokumaneko/items/ff1a1a8020d48299573b
平均二乗誤差によるフィッティング
"""

import numpy as np
from numpy import pi,exp,cos,sin
from matplotlib import pyplot as plt

from transformation import rotation_matrix,rotate

## 回転のみ考える場合

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.grid(False)
ax.axis("off")

# 座標軸
ax.quiver([0,0,0], [0,0,0], [0,0,0], [10,0,0], [0,10,0], [0,0,10], arrow_length_ratio=0.1, color="black")
ax.text(11, 0, 0, "x")
ax.text(0, 11, 0, "y")
ax.text(0, 0, 11, "z")

# トーラス
r = 5
a = 1
theta,phi = np.meshgrid(np.linspace(0,2*pi,10), np.linspace(0,2*pi,20))

x = (r + a*cos(theta))*cos(phi)
y = (r + a*cos(theta))*sin(phi)
z = a*sin(theta)
p = np.vstack((x.flatten(),y.flatten(),z.flatten()))

ax.scatter3D(p[0], p[1], p[2])
# ax.set_box_aspect((np.ptp(p[0]), np.ptp(p[1]), np.ptp(p[2])))

R = rotation_matrix(pi/6, pi/4, pi/2)
pp = rotate(R, p)
ax.scatter3D(pp[0], pp[1], pp[2])
ax.set_box_aspect((np.ptp(pp[0]), np.ptp(pp[1]), np.ptp(pp[2])))

H = np.zeros((3,3))
for i in range(p.shape[1]):
    H += np.matmul(p[:,i].reshape((3,1)), pp[:,i].reshape((1,3)))

u,s,vh = np.linalg.svd(H)
X = np.matmul(vh.transpose(), u.transpose())
print(R - X)

# R = rotation_matrix(pi/6, pi/4, pi/2)
roll = np.arctan2(X[2][1], X[2][2])
pitch = -np.arcsin(X[2][0])
yaw = np.arctan2(X[1][0], X[0][0])
print(roll, pi/6)
print(pitch, pi/4)
print(yaw, pi/2)

plt.show()


## 平行移動を含む場合

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.grid(False)
ax.axis("off")

# 座標軸
ax.quiver([0,0,0], [0,0,0], [0,0,0], [10,0,0], [0,10,0], [0,0,10], arrow_length_ratio=0.1, color="black")
ax.text(11, 0, 0, "x")
ax.text(0, 11, 0, "y")
ax.text(0, 0, 11, "z")

# トーラス
x = (r + a*cos(theta))*cos(phi)
y = (r + a*cos(theta))*sin(phi)
z = a*sin(theta)
p = np.vstack((x.flatten(),y.flatten(),z.flatten()))
T = np.array([-1,-1,-1]).reshape((3,1))
p += T

R = rotation_matrix(pi/6, pi/4, pi/2)
pp = rotate(R, p)
T = np.array([2,3,4]).reshape((3,1))
pp += T

ax.scatter3D(p[0], p[1], p[2])
ax.scatter3D(pp[0], pp[1], pp[2])
ax.set_box_aspect((np.ptp(pp[0]), np.ptp(pp[1]), np.ptp(pp[2])))

mu_p = p.mean(axis=1)
p = p - mu_p.reshape((3,1))
mu_pp = pp.mean(axis=1)
pp = pp - mu_pp.reshape((3,1))

H = np.zeros((3,3))
for i in range(p.shape[1]):
    H += np.matmul(p[:,i].reshape((3,1)), pp[:,i].reshape((1,3)))

u,s,vh = np.linalg.svd(H)
X = np.matmul(vh.transpose(), u.transpose())

T = mu_pp.reshape((3,1)) - np.matmul(X, mu_p.reshape((3,1)))
print(T)

plt.show()
