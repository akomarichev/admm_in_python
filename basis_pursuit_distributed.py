from scipy.sparse import rand
import numpy as np

np.random.seed(0)

n = 300
m = 1200

A = np.random.rand(m, n)
x = rand(n, 1, 0.5)
b = A * x

xtrue = x

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

N = size + 0.0

'''
    Original matlab code you can find under this link: http://web.stanford.edu/~boyd/papers/admm/basis_pursuit/basis_pursuit.html

    Code solves the following problem (parallel version):
        minimize norm(x,1)
        subject to Ax=b

    Written by Artem Komarichev
    Mail: fn9241@wayne.edu
'''


A_splitted = np.array_split(A, [m / N * k for k in range(1, size)])
b_splitted = np.array_split(b, [m / N * k for k in range(1, size)])


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
    r = np.zeros((n, 1))

    AAt = np.dot(A, A.T)
    P = np.eye(n) - np.dot(A.T, np.asarray(np.linalg.lstsq(AAt, A))[0])
    q = np.dot(A.T, np.asarray(np.linalg.lstsq(AAt, b))[0])

    if rank == 0:
        print '{0}\t{1}\t\t{2}\t\t{3}\t\t{4}\t{5}'.format('iter', 'r_norm', 'eps_pri', 's_norm', 'eps_dual', 'obj')

    recv = np.zeros((3, 1))
    send = np.zeros((3, 1))

    for k in range(MAX_ITER):
        # u-update
        u = u + (x - z)

        # x - update
        x = np.dot(P, (z - u)) + q

        # z-update with relaxation
        zold = np.copy(z)
        x_hat = alpha * x + (1 - alpha) * zold

        MPI.COMM_WORLD.Allreduce(x_hat + u, z, op=MPI.SUM)

        z = shrinkage(z / N, 1.0 / (rho * N))

        r = x - z

        send[0] = np.dot(r.T, r)
        send[1] = np.dot(x.T, x)
        send[2] = np.dot(u.T, u)

        MPI.COMM_WORLD.Allreduce(send, recv, op=MPI.SUM)

        obj = objective(A, b, x)
        r_norm = np.sqrt(recv[0])
        s_norm = np.sqrt(N) * rho * np.linalg.norm(z - zold, 2)

        eps_pri = np.sqrt(n * N) * ABSTOL   + RELTOL * \
            max(np.sqrt(recv[1]), np.sqrt(N) * np.linalg.norm(z, 2))
        eps_dual = np.sqrt(n * N) * ABSTOL + RELTOL * np.sqrt(recv[2])

        if rank == 0:
            print '{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(k, r_norm, eps_pri,
                                                        s_norm, eps_dual, obj)

        if r_norm < eps_pri and s_norm < eps_dual:
            break


basis_pursuit(A_splitted[rank], b_splitted[rank], 1.0, 1.0)
