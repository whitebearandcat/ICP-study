import numpy as np
from numpy import arccos,pi
from matplotlib import pyplot as plt
import pickle

from transformation import quaternion_rotate,to_quaternion

with open("d_scanlist.pcl", "rb") as f:
    d_scanlist = pickle.load(f)
with open("qu_scanlist.pcl", "rb") as f:
    qu_scanlist = pickle.load(f)
with open("T_scanlist.pcl", "rb") as f:
    T_scanlist = pickle.load(f)
with open("X.pcl", "rb") as f:
    X = pickle.load(f)
with open("P.pcl", "rb") as f:
    P = pickle.load(f)

# with open("d_scanlist.txt", "w") as f:
#     for d in d_scanlist:
#         f.write(f"{d}")
#         f.write("\n")

# qu = np.array(qu_scanlist)
# angle = 2*np.arccos(qu.T[0]) * 180/np.pi
# with open("angle.txt", "w") as f:
#     for a in angle:
#         f.write(f"{a}")
#         f.write("\n")

min_degree = np.argmin(d_scanlist)

qu = np.array(qu_scanlist)
angle = 2*arccos(qu.T[0]) * 180/pi
for i in range(len(angle)):
    if angle[i] > 180:
        angle[i] -= 360

print(f"theta_0 = {min_degree}, d = {d_scanlist[min_degree]}, angle = {angle[min_degree]}, T = {T_scanlist[min_degree]}")

# initialization
P_ = np.zeros(P.shape)
for i in range(P.shape[1]):
    qu = to_quaternion([0,0,1], min_degree*np.pi/180)
    P[:,i] = quaternion_rotate(P[:,i], qu)
    P[:,i] = quaternion_rotate(P[:,i], qu_scanlist[min_degree]) + T_scanlist[min_degree]

plt.subplot(121)
ax = plt.gca()
ax.plot(d_scanlist)

ax.set_ylim([0,0.05])
ax.set_xlabel(r"$\theta_0$")
ax.set_ylabel("mean square error")
plt.subplot(122)
ax = plt.gca()
ax.set_xlabel(r"$\theta_0$")
ax.set_ylabel("registration angle [deg]")
ax.set_ylim([-180,180])
ax.plot(angle, color="orange")
ax.plot([300,300], [-150,100], color="black", linewidth=1, linestyle="dashed")
ax.plot([-10,360], [0,0], color="black", linewidth=1)
plt.tight_layout()

plt.show()
