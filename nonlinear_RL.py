'''
this code is an implementation of RL for 1-d nonlinear dynamics system
'''

__author__ = "Wooden_Jin"
__copyright__ = "@ZJU_XMECH"

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from math import sin, pi, sqrt

# =================================
# pendulum system
g = 10  # gravity acceleration


def pendulum_sys(x, t, a, b, c, d, e):
    x1, x2 = x
    # a, b, c, d, e = u_param
    u = a * x1 ** 2 + b * x2 ** 2 + c * x1 * x2 + d * x1 + e * x2  # poly2 control law
    temp1 = x2
    temp2 = g * sin(x1) + u
    return np.array([temp1, temp2])


# ==================================
# hyper parameter definition
q1 = 1
q2 = 1
r = 0.01


def fun_loss(parameter):
    """
    this function is the lost function to be optimized
    :param parameter: the control parameter of control law, u=u(x;para)
    :return: total loss value
    """
    total_loss = 0  # total loss of all sampling paths
    loss = []

    # in this situation, we use randomly sample
    # for x1 in (np.random.random(4) - 0.5) * 2 * pi:
    #     for x2 in (np.random.random(4) - 0.5) * 2 * 2 * 2 * sqrt(g):
    for x1 in np.linspace(-pi, pi, 4):
        for x2 in np.linspace(-4 * sqrt(g), 4 * sqrt(g), 6):
            x0 = np.array([x1, x2])  # initial condition
            t = np.arange(0, 10, 0.02)
            track = odeint(pendulum_sys, x0, t, args=(parameter[0],
                                                      parameter[1],
                                                      parameter[2],
                                                      parameter[3],
                                                      parameter[4]))
            u = parameter[0] * np.power(track[:, 0], 2) + \
                parameter[1] * np.power(track[:, 1], 2) + \
                parameter[2] * track[:, 0] * track[:, 1] + \
                parameter[3] * track[:, 0] + \
                parameter[4] * track[:, 1]
            loss.append(q1 * np.sum(np.power(track[:, 0], 2)) + \
                        q2 * np.sum(np.power(track[:, 1], 2)) + \
                        r * np.sum(np.power(u, 2)))
            pass
        pass
    total_loss = sum(loss)
    return total_loss


k0 = (0, 0, 0, -1, -1)
res = minimize(fun_loss, k0, method='SLSQP', options={'maxiter': 10000})
print(res)

# ---------------------------------------------
# phase
fig = plt.figure(figsize=(8, 6))
num_steps = 11
Y, X = np.mgrid[-4 * sqrt(g):4 * sqrt(g):(num_steps * 1j), -pi:pi:(num_steps * 1j)]
U = Y
V = res.x[0] * X * X + res.x[1] * Y * Y + res.x[2] * X * Y + res.x[3] * X + res.x[4] * Y + g * np.sin(X)
speed = np.sqrt(U ** 2 + V ** 2)
plt.streamplot(X, Y, U, V, color=speed, cmap='PuBu', density=2)

for x1 in np.linspace(-pi, pi, 4):
    for x2 in np.linspace(-4 * sqrt(g), 4 * sqrt(g), 6):
        plt.scatter(x1, x2, marker='o', color='deeppink', s=30)
        x0 = np.array([x1, x2])
        K = np.array([[res.x[0], res.x[1]]])
        t = np.arange(0, 1, 0.001)  # simulation time
        track = odeint(pendulum_sys, x0, t, args=(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]))
        plt.scatter(track[:, 0], track[:, 1], s=2, c=cm.Spectral(t))
        pass
    pass

# plt.xlim([-5, 5])
# plt.ylim([-5, 5])
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('./nonlinear_rl.png')
plt.show()

# ================
# original phase
fig = plt.figure(figsize=(8, 6))
num_steps = 11
Y, X = np.mgrid[-4 * sqrt(g):4 * sqrt(g):(num_steps * 1j), -pi:pi:(num_steps * 1j)]
U = Y
V = g*np.sin(X)
speed = np.sqrt(U ** 2 + V ** 2)
plt.streamplot(X, Y, U, V, color=speed, cmap='PuBu', density=2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('./pendulum_phase.png')
plt.show()