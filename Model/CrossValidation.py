import numpy as np

class CrossValidation:

    def __init__(self, model, X, Y, nb_folds=5):
        folds_size = len(X) // nb_folds
        self.all_accuracies = []
        for index in range(nb_folds):
            X_train = [x for i,x in enumerate(X) if i <= index * folds_size or i > (index+1) * folds_size]
            Y_train = [y for i,y in enumerate(Y) if i <= index * folds_size or i > (index+1) * folds_size]
            X_test = [x for i,x in enumerate(X) if i > index * folds_size and i <= (index+1) * folds_size]
            Y_test = [y for i,y in enumerate(Y) if i > index * folds_size and i <= (index+1) * folds_size]
            model.fit(X_train,Y_train)
            self.all_accuracies.append(model.evaluate(X_test, Y_test))

    def accuracy(self):
        return sum(self.all_accuracies) / len(self.all_accuracies)         