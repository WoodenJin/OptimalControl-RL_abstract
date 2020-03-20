from numpy import matrix, array
from control.matlab import *

dt = 0.1

A = matrix([[0, 1],
            [0, 0]])

B = matrix([[0],
            [1]])

Q = matrix([[1, 0],
            [0, 1]])

R = 0.01

(K, X, E) = lqr(A, B, Q, R)
print(K)
print(E)
