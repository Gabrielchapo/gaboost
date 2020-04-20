import C_module
import numpy as np

class LinearRegression:

    def __init__(self, normalize=False):
        self.mean = None
        self.sigma = None
        if normalize == False:
            self.normalize = False
        else:
            self.normalize = True

    def fit(self, X, Y):

        # X isn't an unique value
        if type(X) is list or isinstance(X,np.ndarray):

            # X has multiple parameters
            if type(X[0]) is list or isinstance(X[0],np.ndarray):
                if self.normalize == True:
                    X = np.array([np.array(x) for x in X])
                    self.sigma = [np.amax(x) - np.amin(x) for x in zip(*X)]
                    self.mean = [sum(x) / len(X) for x in zip(*X)]
                    X = (X - self.mean) / self.sigma
                X = [list(x) for x in X]

            # X has one parameter
            else:
                if self.normalize == True:
                    self.mean = sum(X) / len(X)
                    self.sigma = np.amax(X) - np.amin(X)
                    X = [[x] for x in X]
                    X = (X - self.mean) / self.sigma
                X = X.tolist()
                

        # X is an unique value
        else:
            X = [X]
        Y = [[y] for y in Y]
        Y = list(Y)
        self.weights = C_module.regression_fit(X, Y, 0)
    
    def predict(self, X):

        # X isn't an unique value
        if type(X) is list or isinstance(X,np.ndarray):

            # X has multiple parameters
            if len(self.weights) > 2:
                if type(X[0]) is list or isinstance(X[0],np.ndarray):
                    X = np.array([np.array(x) for x in X])
                    if self.normalize == True:
                        X = (X - self.mean) / self.sigma
                    X = [list(x) for x in X]
                else:
                    X = np.array([np.array(x) for x in X])
                    if self.normalize == True:
                        X = (X - self.mean) / self.sigma
                    X = [list(X)]

            # X has one parameter
            else:
                X = [[x] for x in X]
                if self.normalize == True:
                    X = (X - self.mean) / self.sigma
                X = list(X)
        
        # X is an unique value
        else:
            if self.normalize == True:
                X = [[(X - self.mean) / self.sigma]]

        self.weights = [[x] for x in self.weights]
        return C_module.regression_predict(X, self.weights, 0)
    
    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        predictions = [abs((a - b) / b) for a, b in zip(predictions, Y_test)]
        return sum(predictions) / len(predictions)