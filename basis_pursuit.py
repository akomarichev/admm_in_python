from scipy.sparse import rand
import numpy as np

np.random.seed(0)

'''
    Original matlab code you can find under this link: http://web.stanford.edu/~boyd/papers/admm/basis_pursuit/basis_pursuit.html

    Code solves the following problem:
        minimize norm(x,1)
        subject to Ax=b

    Written by Artem Komarichev 
    Mail: fn9241@wayne.edu
'''

n = 30
m = 10

A = np.random.rand(m, n)
x = rand(n, 1, 0.5)
b = A * x

xtrue = x


def shrinkage(a, kappa):
    return np.maximum(0, (a - kappa)) - np.maximum(0, (-a - kappa))


def objective(A, b, x):
    return np.linalg.norm(x, 1)


def basis_pursuit(A, b, rho, alpha):
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    m, n = A.shape

    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))

    AAt = np.dot(A, A.T)
    P = np.eye(n) - np.dot(A.T, np.asarray(np.linalg.lstsq(AAt, A))[0])
    q = np.dot(A.T, np.asarray(np.linalg.lstsq(AAt, b))[0])

    print '{0}\t{1}\t\t{2}\t\t{3}\t\t{4}\t{5}'.format('iter', 'r_norm', 'eps_pri', 's_norm', 'eps_dual', 'obj')

    for k in range(MAX_ITER):
        x = np.dot(P, (z - u)) + q

        # z-update with relaxation
        zold = z
        x_hat = alpha * x + (1 - alpha) * zold
        z = shrinkage(x_hat + u, 1 / rho)

        u = u + (x_hat - z)

        obj = objective(A, b, x)
        r_norm = np.linalg.norm(x - z, 1)
        s_norm = np.linalg.norm(-rho * (z - zold), 1)

        eps_pri = np.sqrt(n) * ABSTOL + RELTOL * \
            max(np.linalg.norm(x, 1), np.linalg.norm(-z, 1))
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * u, 1)

        print '{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(k, r_norm, eps_pri, s_norm, eps_dual, obj)

        if r_norm < eps_pri and s_norm < eps_dual:
            break


basis_pursuit(A, b, 1.0, 1.0)
