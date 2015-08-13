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

m, n = X.shape

_lambda = 0.1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def f(x, *args):
    X, b, z, u, rho, _lambda = args
    m, n = X.shape

    x = np.expand_dims(x, axis=1)

    h = sigmoid(np.dot(X, x))

    return (1.0 / m) * np.sum(-b * np.log(h) - (1 - b) * np.log(1 - h)) + \
        (_lambda / (2 * m)) * np.linalg.norm(x - z + u, 2) ** 2


def gF(x, *args):

    X, b, z, u, rho, _lambda = args
    x = np.expand_dims(x, axis=1)

    h = sigmoid(np.dot(X, x))

    m = X.shape[0]

    out = (1.0 / m) * (np.dot(X.T, (h - b)) + _lambda * (x - z + u))

    return np.asarray(out, order='F')


def shrinkage(a, kappa):
    # print 'a-kappa:', a - kappa
    return np.maximum(0, (a - kappa)) - np.maximum(0, (-a - kappa))


def objective(X, b, x, z, u, _lambda):
    m = X.shape[0]
    h = sigmoid(np.dot(X, x))
    return (1.0 / m) * np.sum(-b * np.log(h) - (1 - b) * np.log(1 - h)) + \
        (_lambda / (2 * m)) * np.linalg.norm(z, 2) ** 2


def logreg(A, b, _lambda, rho, alpha):
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    m, n = A.shape

    x = np.zeros((n + 1, 1))
    z = np.zeros((n + 1, 1))
    u = np.zeros((n + 1, 1))

    X = np.hstack((np.ones((m, 1)), A))

    # print '{0}\t{1}\t\t{2}\t\t{3}\t\t{4}\t{5}'.format('iter', 'r_norm',
    #                                                   'eps_pri', 's_norm', 'eps_dual', 'obj')

    for k in range(MAX_ITER):
        # x-update

        #print(x - z + u).shape
        res = minimize(f,
                       x,
                       # fprime=gF,
                       jac=gF,
                       args=(X, b, z, u, rho, _lambda),
                       # approx_grad=True,
                       # epsilon=0.5
                       method='L-BFGS-B',
                       # method='CG',
                       # jac = gF,
                       # maxiter=25
                       options={'maxiter': 15}
                       )
        # print res
        #x = np.expand_dims(res[0], axis=1)
        x = np.expand_dims(res.x, axis=1)

        # z-update
        zold = np.copy(z)
        x_hat = alpha * x + (1 - alpha) * zold
        # print 'x_hat', x_hat.T
        z = shrinkage(x_hat + u, (m * 0.0001) / rho)

        # u-update
        u = u + (x_hat - z)

        obj = objective(X, b, x, z, u, _lambda)
        r_norm = np.linalg.norm(x - z, 2)
        s_norm = np.linalg.norm(rho * (z - zold), 2)

        eps_pri = np.sqrt(n) * ABSTOL + RELTOL * \
            max(np.linalg.norm(x, 2), np.linalg.norm(z, 2))
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * u, 2)

        # print '{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(k, r_norm, eps_pri,
        #                                             s_norm, eps_dual, obj)

        if r_norm < eps_pri and s_norm < eps_dual:
            break

    return x

# random_indecies = np.random.randint(0, y.shape[0], 5000)


all_theta = np.zeros((num_labels, n + 1))

print 'Predictions for numbers: '
for c in range(0, num_labels):
    if c == 0:
        theta = logreg(X, (y == 10) + 0, _lambda, 1.0, 1.0)
    else:
        theta = logreg(X, (y == c) + 0, _lambda, 1.0, 1.0)

    all_theta[c, :] = theta.T

    print 'c: ', c

X = np.hstack((np.ones((m, 1)), X))
predictions = np.dot(all_theta, X.T)


maxval = np.amax(predictions, axis=0)
maxindices = np.argmax(predictions, axis=0)

maxindices[maxindices == 0] = 10

maxindices = np.expand_dims(maxindices, axis=1)

print 'accuracy: ', 100.0 * np.sum((y == maxindices) + 0) / 5000.0, '%'

# logreg(
#     X[random_indecies], (y[random_indecies] == 1) + 0, _lambda, 1.0, 1.0)
