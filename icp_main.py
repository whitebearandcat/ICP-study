"""
点群のレジストレーションの初歩
https://qiita.com/shirokumaneko/items/ff1a1a8020d48299573b
イテレーション
"""

import numpy as np
from numpy import pi
import random
import pickle

from transformation import quaternion_rotate, rotation_matrix,to_quaternion
from icp import icp

X = np.zeros((3,1))
for x_i in np.arange(0,2,0.05):
    for y_i in np.arange(0,1,0.05):
        if y_i <= -x_i/2 + 1:
            p_i = np.array([[x_i], [y_i], [0]])
            X = np.hstack((X, p_i))

mu_x = X.mean(axis=1)
X = X - mu_x.reshape((3,1))

R = rotation_matrix(0, 0, pi/3)
P = np.matmul(R, X)
mu_p = P.mean(axis=1)
P = P - mu_p.reshape((3,1))
for i in range(P.shape[1]):
    P[0][i] += random.uniform(-0.01, 0.01)
    P[1][i] += random.uniform(-0.01, 0.01)

d_scanlist = []
qu_scanlist = []
T_scanlist = []

scan_step = pi/180 # 1°

for theta in np.arange(0,2*pi,scan_step):
    # initialization
    P_0 = np.zeros(P.shape)
    qu = to_quaternion([0,0,1], theta)
    for i in range(P.shape[1]):
        P_0[:,i] = quaternion_rotate(P[:,i], qu)

    # iteration
    qu,T,d_k = icp(P_0, X)

    print(f"theta = {theta*180/pi}: d_k = {d_k}")
    d_scanlist.append(d_k)
    qu_scanlist.append(qu)
    T_scanlist.append(T)

min_degree = np.argmin(d_scanlist)
print(min_degree, d_scanlist[min_degree])

with open("d_scanlist.pcl", "wb") as f:
    pickle.dump(d_scanlist, f)
with open("qu_scanlist.pcl", "wb") as f:
    pickle.dump(qu_scanlist, f)
with open("T_scanlist.pcl", "wb") as f:
    pickle.dump(T_scanlist, f)
with open("X.pcl", "wb") as f:
    pickle.dump(X, f)
with open("P.pcl", "wb") as f:
    pickle.dump(P, f)
