from pyMPC.mpc import MPCController
import numpy as np
import scipy.sparse as sparse

dt = 0.01

Ad = np.array([[1, dt], [0, 1]])
Bd = np.array([[0], [dt]])

xref = np.array([0, 0])
uref = np.array([0])

xmin = np.array([-100.0, -100.0])
xmax = np.array([100.0, 100.0])

umin = np.array([-1000])
umax = np.array([1000])

Dumin = np.array([-1000])
Dumax = np.array([1000])

Qx = sparse.diags([1, 1])
QxN = sparse.diags([1, 1])

Qu = 0.01 * sparse.eye(1)
QDu = 0.0 * sparse.eye(1)

x0 = np.array([1, 0])

uminus1 = np.array([0.0])

Np = 1000

K = MPCController(Ad, Bd, Np=Np, x0=x0, xref=xref, uminus1=uminus1,
                  Qx=Qx, QxN=QxN, Qu=Qu, QDu=QDu,
                  xmin=xmin, xmax=xmax, umin=umin, umax=umax, Dumin=Dumin, Dumax=Dumax)
K.setup()
uMPC = K.output()
# uMPC = K.output(return_u_seq=True)
print(uMPC)