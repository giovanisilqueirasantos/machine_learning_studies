import numpy as np

class AdalineSGD(object):
    """
    Adaptive Linear Neuron classfier with Stochastic Gradient Descent.

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
    shuffle : bool (default: true)
        Shuffles training data every epoch.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)

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

        self._initialize_wights(X.shape[1])
        self._cost = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self._cost.append(avg_cost)
        
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_wights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_wights(self, m):
        """Initialize weights to zeros"""
        self._w = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline Learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self._w[1:] += self.eta * xi.dot(error)
        self._w[0] += self.eta * error
        cost = .5 * error**2

        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self._w[1:] + self._w[0])

    def activation(self, X):
        """Comput linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)