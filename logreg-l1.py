from scipy.sparse import rand, spdiags
import numpy as np

np.random.seed(0)

'''
    Original matlab code you can find under this link: http://web.stanford.edu/~boyd/papers/admm/logreg-l1/logreg.html

    Code solves the following problem:
        minimize   sum( log(1 + exp(-b_i*(a_i'w + v)) ) + m*mu*norm(w,1)

    Written by Artem Komarichev
    Mail: fn9241@wayne.edu
'''

n = 50
m = 200

w = rand(n, 1, 0.1)
v = np.random.randn(1)

X = rand(m, n, 10.0 / n)
btrue = np.sign(X * w - v)

# noizy b
b = np.sign(X * w - v + np.sqrt(0.1) * np.random.rand(m, 1))

A = spdiags(b.T, 0, m, m) * X

ratio = sum(b == 1) / (m + 0.0)

mu = 0.1 * 1 / (m + 0.0) * np.linalg.norm((1 - ratio) * np.sum(A[np.repeat(
    b == 1, 50, axis=1)].toarray()) + ratio * np.sum(A[np.repeat(b == -1, 50, axis=1)].toarray()), np.inf)


def logreg(A, b, mu, rho, alpha):
    pass

logreg(A, b, mu, 1.0, 1.0)
