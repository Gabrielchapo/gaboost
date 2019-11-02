import numpy as np

class LinearRegression:

    def fit(self, X, Y, epochs, gradient):
        self.thetas = np.zeros((X.shape[1], 1))
        m = X.shape[0]

        for i in range(epochs):
            predicted = np.dot(X, self.thetas)
            diff = predicted - Y
            gradient_vector = np.dot(X.T, diff)
            self.thetas -= (gradient / m) * gradient_vector
            print("Epoch:", i + 1, "/", epochs, "=== Loss:", np.sum((diff ** 2)) / (2 * m))

    def predict(self, X):
        return np.dot(X, self.thetas)