import numpy as np

class AdalineGD(object):
    """
    Adaptive Linear Neuron classfier with Gradient Descent.

    Parameters
    ----------
    eta : float
        learning rate (between 0.0 and 1.0)
    n_iter : int
        passes over the training dataset.

    Atrributes
    ----------
    _w : id-array
        Wights after fitting.
    _cost : list
        Number of cost in every iteration.
    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : [array-like], shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : [array-like], shape = [n_samples]
            Target values.

        Returns
        -------
        self: object
        """

        self._w = np.zeros(1 + X.shape[1])
        self._cost = []

        for _ in range(self.n_iter):
            output = self.activation(X)
            errors = (y - output)
            self._w[1:] += self.eta * X.T.dot(errors)
            self._w[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self._cost.append(cost)
        
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self._w[1:] + self._w[0])

    def activation(self, X):
        """Comput linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)