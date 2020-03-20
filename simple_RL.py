'''
this code is an implementation of simple RL example
'''
__author__ = "Wooden_Jin"
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm


# ===============================
# system dynamics
def linear_sys(x, t, A, B, K):
    # system: dot_x = Ax+Bu
    # where: u=Kx
    x1, x2 = x
    temp = (A + B * K)
    temp = np.dot(temp, np.array([[x1], [x2]]))
    return np.array([temp[0, 0], temp[1, 0]])


A = np.array([[0, 1],  # system dynamics matrix
              [0, 0]])
B = np.array([[0], [1]])  # b matrix

# ===============================
# hyper parameter definition
q1 = 1
q2 = 1
r = 0.01


def fun_loss(parameter):
    """
    this function is the lost function to be optimized
    :param parameter: the control parameter of control law, u=u(x;para)
    :return: total loss value of the sampling paths
    """

    total_loss = 0  # total loss of all sampling paths
    loss = []

    # for this simple problem, we use uniform sampling
    for x1 in np.linspace(-3, 3, 4):  # uniform sample position
        for x2 in np.linspace(-3, 3, 4):  # uniform sample velocity

            x0 = np.array([x1, x2])  # initial condition
            K = np.array([[parameter[0], parameter[1]]])  # linear control law
            t = np.arange(0, 10, 0.01)  # simulation time
            track = odeint(linear_sys, x0, t, args=(A, B, K))  # dynamics system simulation
            loss.append(q1 * np.sum(np.power(track[:, 0], 2)) + \
                        q2 * np.sum(np.power(track[:, 1], 2)) + \
                        r * np.sum(np.power(track[:, 0] * parameter[0] + track[:, 1] * parameter[1], 2)))

            pass
        pass
    total_loss = sum(loss)
    return total_loss


k0 = (-1, -1)
res = minimize(fun_loss, k0, method='SLSQP', options={'maxiter': 10000})
print(res)

# ----------------------------------------------------------------------------------------
# ================
# visualization part
# plot phase of the dynamics
fig = plt.figure(figsize=(5, 5))
num_steps = 11
Y, X = np.mgrid[-5:5:(num_steps * 1j), -5:5:(num_steps * 1j)]
U = Y
V = res.x[0] * X + res.x[1] * Y
speed = np.sqrt(U ** 2 + V ** 2)
# color = cm.Spectral(speed)
# color[:, :, 3] = 0.5
plt.streamplot(X, Y, U, V, color=speed, cmap='PuBu')

for x1 in np.linspace(-3, 3, 4):
    for x2 in np.linspace(-3, 3, 4):
        plt.scatter(x1, x2, marker='o', color='deeppink', s=30)
        x0 = np.array([x1, x2])
        K = np.array([[res.x[0], res.x[1]]])
        t = np.arange(0, 1, 0.001)  # simulation time
        track = odeint(linear_sys, x0, t, args=(A, B, K))
        plt.scatter(track[:, 0], track[:, 1], s=2, c=cm.Spectral(t))
        pass
    pass

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('./rl_sampling.png')
plt.show()
