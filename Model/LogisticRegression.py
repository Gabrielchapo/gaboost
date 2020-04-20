import C_module
import numpy as np

class LogisticRegression:

    def __init__(self, normalize=False):
        self.mean = None
        self.sigma = None
        if normalize == False:
            self.normalize = False
        else:
            self.normalize = True

    def fit(self, X, Y):
        
        if self.normalize == True:
            X = np.array([np.array(x) for x in X])
            self.sigma = [np.amax(x) - np.amin(x) for x in zip(*X)]
            self.mean = [sum(x) / len(X) for x in zip(*X)]
            X = (X - self.mean) / self.sigma
            X = [list(x) for x in X]

        self.weights = C_module.regression_fit(X, Y, 1)

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

        return C_module.regression_predict(X, self.weights, 1)
        
    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        predictions = [np.argmax(x) for x in predictions]
        Y_test = [np.argmax(x) for x in Y_test]
        count = 0
        for i in range(len(Y_test)):
            if Y_test[i] == predictions[i]:
                count += 1
        return count / len(Y_test)
        