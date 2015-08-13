from scipy.sparse import rand, spdiags
from scipy.optimize import minimize, fmin_l_bfgs_b, fmin_cg
import numpy as np
import scipy.io

'''
    Original matlab code you can find under this link: http://web.stanford.edu/~boyd/papers/admm/logreg-l1/logreg.html

    Code solves the following problem:
        minimize   sum( log(1 + exp(-b_i*(a_i'w + v)) ) + m*mu*norm(w,1)

    Written by Artem Komarichev
    Mail: fn9241@wayne.edu
'''

input_layer_size = 400
num_labels = 10

data = scipy.io.loadmat('ex3data.mat')
X = data['X']
y = data['y']

print type(X), X.shape
print type(y), y.shape

_lambda = 0.1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def f(x, *args):
    X, b, z, u, rho, _lambda = args
    m, n = X.shape

    x = np.expand_dims(x, axis=1)

    # print x.shape
    # print X.shape
    # print x.shape
    # print "OK!!!"
    h = sigmoid(np.dot(X, x))
    # print "OK!!!"
    # print type(x)

    #print -b.T * np.log(h) - (1 - b.T) * np.log(1 - h)
    # print z.shape
    # print type(x)
    # print u.shape
    # print(x - z.T + u.T).shape

    # print(1.0 / m) * np.sum(-b * np.log(h) - (1 - b) * np.log(1 - h)) + \
    #     (_lambda / (2 * m)) * np.linalg.norm(x - z + u, 2) ** 2

    return (1.0 / m) * np.sum(-b * np.log(h) - (1 - b) * np.log(1 - h)) + \
        (_lambda / (2 * m)) * np.linalg.norm(x - z + u, 2) ** 2


def gF(x, *args):

    X, b, z, u, rho, _lambda = args
    x = np.expand_dims(x, axis=1)
    # print 'ok!'
    h = sigmoid(np.dot(X, x))

    m = X.shape[0]

# print X.T.shape
# print(h - b.T).T.shape

    #print(np.dot(X.T, (h - b.T).T)).shape
    # print np.asarray(x).shape
    # print z.shape
    # print u.shape
    # print(x - z + u).shape

    # print type(h - b)
    # print(h - b).shape
    # print X.T.shape
    # print type(X.T)
    #print(x - z + u).shape

    out = (1.0 / m) * (np.dot(X.T, (h - b)) + _lambda * (x - z + u))
    # print 'OK!'
    # print "gF:"
    # print x.shape
    # print type(x)
    # print out.shape
    # print type(out)

    return np.asarray(out, order='F')


def shrinkage(a, kappa):
    # print 'a-kappa:', a - kappa
    return np.maximum(0, (a - kappa)) - np.maximum(0, (-a - kappa))


# def objective(A, b, mu, x, z):
#     m = A.shape[0]
# return np.sum(np.log(1 + np.exp(-A * x[1:] - b * x[0]))) + m * mu *
# np.linalg.norm(z, 1)


def logreg(A, b, _lambda, rho, alpha):
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    m, n = A.shape

    x = np.zeros((n + 1, 1))
    z = np.zeros((n + 1, 1))
    u = np.zeros((n + 1, 1))

    X = np.hstack((np.ones((m, 1)), A))
    # C = np.concatenate((b, A), axis=0)

    print '{0}\t{1}\t\t{2}\t\t{3}\t\t{4}\t{5}'.format('iter', 'r_norm', 'eps_pri', 's_norm', 'eps_dual', 'obj')

    for k in range(MAX_ITER):
        # x-update

        #print(x - z + u).shape
        res = fmin_l_bfgs_b(f,
                            np.zeros((n + 1, 1)),
                            fprime=gF,
                            # jac=gF,
                            args=(X, b, z, u, rho, _lambda),
                            # approx_grad=True,
                            # epsilon=0.5
                            # method='L-BFGS-B',
                            # method='CG',
                            # jac = gF,
                            maxiter=25
                            #options={'maxiter': 5, 'maxfun': 5}
                            )
        # print res
        x = np.expand_dims(res[0], axis=1)
        #x = np.expand_dims(res.x, axis=1)
        # print 'ok'
        # print res
        # print res.shape
        # x = res[0]
        # print x.shape
        obj = 0
        # x = update_x(A, b, u, z, rho)

        # z-update
        zold = np.copy(z)
        x_hat = alpha * x + (1 - alpha) * zold
        # print 'x_hat', x_hat.T
        z = shrinkage(x_hat + u, (m * 0.0009) / rho)

        # u-update
        u = u + (x_hat - z)

        # print 'z:', z.T
        # print 'zold:', zold.T

        #obj = objective(A, b, mu, x, z)
        r_norm = np.linalg.norm(x - z, 2)
        s_norm = np.linalg.norm(rho * (z - zold), 2)

        eps_pri = np.sqrt(n) * ABSTOL + RELTOL * \
            max(np.linalg.norm(x, 2), np.linalg.norm(z, 2))
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * u, 2)

        print '{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(k, r_norm, eps_pri,
                                                    s_norm, eps_dual, obj)

        if r_norm < eps_pri and s_norm < eps_dual:
            break

random_indecies = np.random.randint(0, y.shape[0], 5000)

#logreg(X, (y == 1) + 0, _lambda, 1.0, 1.0)

logreg(
    X[random_indecies], (y[random_indecies] == 1) + 0, _lambda, 1.0, 1.0)
