import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.num_classes = 3
        self.mu = None
        self.sigma = None
    # Just to show how to make 'private' methods
    def __dummy_private_method(self, input):
        return None

    def calc_mu(self, N, X, y):
        mu = [0 for i in range(self.num_classes)]
        cnt = [0 for i in range(self.num_classes)]
        for i in range(N):
            mu[y[i]] += X[i]
            cnt[y[i]] += 1
        a = np.asmatrix(cnt).T
        return np.divide(mu, a)

    def calc_sigma_same(self, N, X, y, mu):
        s = (X[0] - mu[y[0]]).T * (X[0] - mu[y[0]])
        mat = [np.zeros(s.shape) for i in range(self.num_classes)]
        for i in range(N):
            mat[y[i]] += (X[i] - mu[y[i]]).T * (X[i] - mu[y[i]])
        return mat

    # TODO: Implement this method!
    def fit(self, X, y):
        N = X.shape[0]
        self.mu = self.calc_mu(N, X, y)
        mat = self.calc_sigma_same(N, X, y, self.mu)
        if self.is_shared_covariance:
            self.sigma = [sum(mat) for i in range(self.num_classes)]
        else:
            self.sigma = mat

    # TODO: Implement this method!
    def predict(self, X_pred):
        from scipy.stats import multivariate_normal
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            vals = []
            for i in range(self.num_classes):
                a = np.ndarray.tolist(self.mu[i])[0]
                if self.is_shared_covariance:
                    cov = self.sigma[i]
                else:
                    cov = self.sigma[i]
                vals.append(multivariate_normal.pdf(x, mean=a, cov=cov))
            preds.append(vals.index(max(vals)))
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        # preds = [(y == j).astype(int) for j in range(self.num_classes)]
        # corr_mu = np.asarray([self.mu[i].tolist()[0] for i in y])
        # l = np.dot((X - corr_mu), np.linalg.inv(self.sigma))
        # l = np.subtract(np.dot(l, (X - corr_mu).T), np.log(np.linalg.det(self.sigma)))
        N = X.shape[0]
        cnt = 0
        for i in range(N):
            for j in range(self.num_classes):
                if y[i] == j:
                    cnt += (X[i] - self.mu[j]) * np.linalg.inv(self.sigma[j]) * (X[i] - self.mu[j]).T - np.log(np.linalg.det(self.sigma[j]))
        return cnt[0]/2
