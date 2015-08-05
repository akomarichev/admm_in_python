from scipy.sparse import rand
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
