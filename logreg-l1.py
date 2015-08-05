from scipy.sparse import rand, spdiags, hstack
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
    b == 1, n, axis=1)].toarray()) + ratio * np.sum(A[np.repeat(b == -1, n, axis=1)].toarray()), np.inf)


def f(x, C, z, u, rho):
    return np.sum(np.log(1 + np.exp(C * x))) + (rho / 2) * np.linalg.norm(x - z + u, 2) ** 2


def update_x(A, b, u, z, rho):
    alpha = 0.1
    BETA = 0.5
    TOLERANCE = 1e-5
    MAX_ITER = 50

    m, n = A.shape
    I = np.eye(n + 1)
    x = np.zeros((n + 1, 1))
    C = hstack((-b, -A))

    for i in range(MAX_ITER):
        fx = f(x, C, z, u, rho)
        g = C.T * (np.exp(C * x) / (1 + np.exp(C * x))) + rho * (x - z + u)
        L = np.diag(np.exp(C * x) / (1 + np.exp(C * x)) ** 2)
        H = C.T * np.diag(np.repeat(L, m, axis=0)) * C + rho * I
        dx = np.asarray(np.linalg.lstsq(-H, g)[0])
        dfx = g.T * dx

        if (np.abs(dfx) < TOLERANCE).all():
            break

        # backtracking
        t = 1
        while (f(x + t * dx, C, z, u, rho) > fx + alpha * t * dfx).all():
            t = BETA * t
        x = x + t * dx

    return x


def shrinkage(a, kappa):
    return np.maximum(0, (a - kappa)) - np.maximum(0, (-a - kappa))


def objective(A, b, mu, x, z):
    m = A.shape[0]
    return np.sum(np.log(1 + np.exp(-A * x[1:] - b * x[0]))) + m * mu * np.linalg.norm(z, 1)


def logreg(A, b, mu, rho, alpha):
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    m, n = A.shape

    x = np.zeros((n + 1, 1))
    z = np.zeros((n + 1, 1))
    u = np.zeros((n + 1, 1))

    print '{0}\t{1}\t\t{2}\t\t{3}\t\t{4}\t{5}'.format('iter', 'r_norm', 'eps_pri', 's_norm', 'eps_dual', 'obj')

    for k in range(MAX_ITER):
        # x-update
        x = update_x(A, b, u, z, rho)

        # z-update
        zold = np.copy(z)
        x_hat = alpha * x + (1 - alpha) * zold
        z = shrinkage(x_hat + u, (mu * m) / rho)

        # u-update
        u = u + (x_hat - z)

        obj = objective(A, b, mu, x, z)
        r_norm = np.linalg.norm(x - z, 2)
        s_norm = np.linalg.norm(rho * (z - zold), 2)

        eps_pri = np.sqrt(n) * ABSTOL + RELTOL * \
            max(np.linalg.norm(x, 2), np.linalg.norm(z, 2))
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * u, 2)

        print '{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(k, r_norm, eps_pri,
                                                    s_norm, eps_dual, obj)

        if r_norm < eps_pri and s_norm < eps_dual:
            break

logreg(A, b, mu, 1.0, 1.0)
