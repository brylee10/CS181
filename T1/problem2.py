import math
import numpy as np

def kern(x, x_, W):
    a = (x-x_).reshape(1, 2)
    return math.exp(-a * W * a.T)

# X is a 2 x n matrix of all data points
# x is the input data point
# Y output vector
# i is the current index (i.e. data point) in the X matrix
def pred(x, X, Y, W, i):
    numerator = sum([kern(x, x_, W) * y_ if j != i else 0 for j, (x_, y_) in enumerate(zip(X, Y))])
    denominator = sum([kern(x, x_, W) if j != i else 0 for j, x_ in enumerate(X)])
    return numerator/denominator

# Returns error with kernel W
def loss(X, Y, W):
    return 0.5 * sum([(y_-pred(x_, X, Y, W, i))**2 for i, (x_, y_) in enumerate(zip(X, Y))])

x1 = [0, 0, 0, .5, .5, .5, 1, 1, 1]
x2 = [0, .5, 1, 0, .5, 1, 0, .5, 1]
X = np.asarray([x1, x2]).T
y = np.asarray([0, 0, 0, .5, .5, .5, 1, 1, 1]).T

alphas = [10, 100, 1000]
kernel_name = ["W1", "W2", "W3"]
for alpha in alphas:
    print("Alpha " + str(alpha))
    #print("Loss W1")
    W1 = np.matrix([[1, 0], [0, 1]]) * alpha
    #print("Loss W2")
    W2 = np.matrix([[0.1, 0], [0, 1]]) * alpha
    #print("Loss W3")
    W3 = np.matrix([[1, 0], [0, 0.1]]) * alpha

    kernels = [W1, W2, W3]
    for k, name in zip(kernels, kernel_name):
        print("Loss " + name)
        print(loss(X, y, k))
    print()