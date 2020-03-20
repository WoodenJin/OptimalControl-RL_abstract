"""
this script is used to simulate linear feedback system
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm


def linear_sys(x, t, A, B, K):
    # system: dot_x = Ax+Bu
    # where: u=Kx
    x1, x2 = x
    temp = (A + B * K)
    temp = np.dot(temp, np.array([[x1], [x2]]))
    return np.array([temp[0, 0], temp[1, 0]])


t = np.arange(0, 10, 0.001)  # simulation time
x0 = np.array([10, 10])  # initial condition
A = np.array([[0, 1],  # system dynamics matrix
              [0, 0]])
B = np.array([[0], [1]])  # b matrix
K = np.array([[-2, -3]])  # control law

track = odeint(linear_sys, x0, t, args=(A, B, K))

fig = plt.figure(figsize=(10, 5))
plt.style.use('seaborn-deep')
# temp = cm.winter(t / 10)
# plt.plot(track[:, 0], track[:, 1], lw=3, c=cm.hot(t / 10))
plt.subplot(1, 2, 1)
plt.scatter(track[:, 0], track[:, 1], linewidths=0.5, c=cm.Spectral(t / 5))
plt.xlabel('x1')
plt.ylabel('x2')

plt.subplot(1, 2, 2)
plt.plot(t, track[:, 0], lw=3, label='x1')
plt.plot(t, track[:, 1], lw=3, label='x2')
plt.xlabel('t')
plt.legend()
# plt.savefig('./linear_fdc.png')
plt.show()

# ==========================
# plot the phase of the dynamics system
num_steps = 11
Y, X = np.mgrid[-25:25:(num_steps * 1j), -25:25:(num_steps * 1j)]
U = Y
V = -2 * X + -3 * Y
speed = np.sqrt(U ** 2 + V ** 2)
plt.streamplot(X, Y, U, V, color=speed)
plt.scatter(track[:, 0], track[:, 1], linewidths=0.5, c=cm.Spectral(t / 5))
# plt.savefig('./phase.png')
plt.show()
