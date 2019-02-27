import numpy as np


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
        self.W = None
        self.log_likelihood_vals = []

    # Just to show how to make 'private' methods
    def __dummy_private_method(self, input):
        return None

    # Return derivative of log likelihood wrt w_j (class j)
    def deriv_neg_log(self, W, x, y, j):
        error = 0
        val = 1 if y == j else 0
        # Make X proper dimensions and turn to matrix
        a = np.asmatrix(x)
        a = a.transpose()
        error += (self.stablesoftmax(W, a)[j] - val) * x
        error = np.asmatrix(error) + self.lambda_parameter/2 * W[j]
        return error

    # Returns a numerically stable softmax of W * x
    def stablesoftmax(self, W, x):
        """Compute the softmax of vector x in a numerically stable way."""
        a = np.dot(W, x.T)
        shiftx = a - np.max(a)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def calc_log_likelihood(self, W, X, y, N):
        likelihood = 0
        for i in range(N):
            # Make X proper dimensions and turn to matrix
            a = np.asmatrix(X[i])
            likelihood += np.log(self.stablesoftmax(W, a)[y[i]])
        likelihood = np.asmatrix(likelihood)
        return likelihood

    # TODO: Implement this method!
    def fit(self, X, y):
        num_classes = 3
        change_limit = 0.001
        N = X.shape[1]
        self.W = np.random.rand(num_classes, N)
        cnt = 0
        while True:
            stop = False
            # for j in range(num_classes):
            #     for i in range(N):
            #         change = self.eta * self.deriv_neg_log(self.W, X[i], y[i], j)
            #         self.W[j] = self.W[j] - change
            #         if (change > change_limit).any():
            #             stop = False
            pred = [(y == j).astype(int) for j in range(num_classes)]
            change = np.dot(pred - self.stablesoftmax(self.W, X), X) + self.lambda_parameter * self.W
            self.W = self.W - self.eta * change
            if (change > change_limit).any():
                stop = False
            # Convert from matrix to number via .item()
            self.log_likelihood_vals.append(self.calc_log_likelihood(self.W, X, y, N).item(0))
            if stop or cnt > 50000:
                break
            cnt += 1
            if cnt % 1000 == 0:
                print("Iteration " + str(cnt))
        print(self.W)
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        preds = []
        for x in X_pred:
            preds.append(self.stablesoftmax(self.W, x).argmax())
        return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.log_likelihood_vals)), self.log_likelihood_vals)
        plt.title('Neg Log Likelihood vs Iterations: ' + 'Lambda = ' + str(self.lambda_parameter) + \
                  ' Rate = ' + str(self.eta))
