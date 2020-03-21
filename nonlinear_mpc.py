'''
this code is an implementation of RL for 1-d nonlinear dynamics system, using NN as base function
reference: https://www.do-mpc.com/en/latest/getting_started.html
'''

__author__ = "Wooden_Jin"
__copyright__ = "@ZJU_XMECH"

import numpy as np
import sys
import do_mpc
from casadi import *  # symbolic library CasADi

model_type = 'continuous'  # define model type
model = do_mpc.model.Model(model_type)  # create model object

"""
dynamics model:
x1_dot = x2
x2_dot = gsin(x1)/l2 + u/ml2
"""
# set variable of the dynamics system
theta = model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
# x2 = model.set_variable(var_type='_x', var_name='theta_dot', shape=(1, 1))
d_theta = model.set_variable(var_type='_x', var_name='d_theta', shape=(1, 1))
u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))

# l = model.set_variable(var_type='_p', var_name='l')  # pendulum length
# m = model.set_variable(var_type='_p', var_name='m')  # mass length
l = 1
m = 1

model.set_rhs('theta', d_theta[0])

d_theta_next = vertcat(
    sin(theta) * 10 / l / l + u / m / l / l
)

model.set_rhs('d_theta', d_theta_next)

model.setup_model()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 500,
    't_step': 0.02,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)
q1 = 1
q2 = 1
r = 0.01
mterm = q1 * theta ** 2 + q2 * d_theta ** 2
lterm = q1 * theta ** 2 + q2 * d_theta ** 2 + r * u ** 2

mpc.set_objective(mterm=mterm, lterm=lterm)

# rterm = r * u ** 2
mpc.set_rterm(u=1e-2)

mpc.bounds['lower', '_x', 'theta'] = -np.pi
mpc.bounds['upper', '_x', 'theta'] = np.pi

mpc.bounds['lower', '_u', 'u'] = -100
mpc.bounds['upper', '_u', 'u'] = 100

l_ = 1 * 1e-4 * np.array([1., 0.9, 1.1])
m_ = 1 * 1e-4 * np.array([1., 0.9, 1.1])
# mpc.set_uncertainty_values([l_, m_])
mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=0.02)

p_template = simulator.get_p_template()


def p_fun(t_now):
    # p_template['l'] = 1
    # p_template['m'] = 1
    return p_template


simulator.set_p_fun(p_fun)
simulator.setup()

x0 = np.array([0.5, 10]).reshape(-1, 1)
simulator.set_initial_state(x0, reset_history=True)
mpc.set_initial_state(x0, reset_history=True)

# =======================================================
# visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True
graphics = do_mpc.graphics.Graphics()
fig, ax = plt.subplots(2, sharex=True, figsize=(16, 9))
fig.align_ylabels()
graphics.add_line(var_type='_x', var_name='theta', axis=ax[0])
graphics.add_line(var_type='_x', var_name='d_theta', axis=ax[0])
ax[0].set_ylabel('state')

graphics.add_line(var_type='_u', var_name='u', axis=ax[1])
ax[1].set_ylabel('control')
ax[1].set_xlabel('time [s]')

u0 = np.zeros((1, 1))
data = []
data.append([])
data.append([])
data.append([])
for i in range(2000):
    x = simulator.make_step(u0)
    u0 = mpc.make_step(x)
    data[0].append(x[0, 0])
    data[1].append(x[1, 0])
    data[2].append(u0[0, 0])
    print("x={},u={}".format(x, u0))
    # simulator.make_step(u0)
    # u0 = mpc.make_step(x0)
graphics.plot_results(simulator.data)
plt.show()

# print(data)

# compare the trajectory with nonlinear-RL
# phase
plt.style.use('seaborn-deep')
fig = plt.figure(figsize=(8, 6))
num_steps = 11
g = 10
res = [-0.067, 0.00078, -0.031, -15.41, -9.28]
Y, X = np.mgrid[-4 * sqrt(g):4 * sqrt(g):(num_steps * 1j), -pi:pi:(num_steps * 1j)]
U = Y
V = res[0] * X * X + res[1] * Y * Y + res[2] * X * Y + res[3] * X + res[4] * Y + g * np.sin(X)
speed = np.sqrt(U ** 2 + V ** 2)
plt.streamplot(X, Y, U, V, color=speed, cmap='PuBu', density=2)
plt.scatter(data[0], data[1], s=30, c=cm.Spectral(np.linspace(0, len(data[0]) * 0.1, len(data[0])) / 40))
plt.plot(data[0], data[1], color='C2', lw=2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-pi, pi])
plt.ylim([-4 * sqrt(g), 4 * sqrt(g)])
plt.savefig('./nonlinear_mpc.png')
plt.show()
