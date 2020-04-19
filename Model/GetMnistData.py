import numpy as np
import json

class GetMnistData:
    
    def __init__(self, train_path, test_path):

        ## Load and prepare data from mnist_handwritten_train.json
        try:
            with open(train_path, 'r') as file:
                content = file.read()
                file.close()
        except:
            exit("Error: mnist_handwritten_train.json not found")
        content = json.loads(content)
        try:
            X = []
            Y = []
            for x in content:
                X.append(x["image"][:])
                Y.append(x["label"])
        except:
            exit("Error: incorrect JSON format")
        self.X_train = np.array(X, float)
        self.Y_train = np.zeros((self.X_train.shape[0], 10), float)
        # preparing one-hot label Y train
        i = 0
        for y in Y:
            self.Y_train[i][y] = 1
            i += 1

        ## Load and prepare data from mnist_handwritten_test.json
        try:
            with open(test_path, 'r') as file:
                content = file.read()
                file.close()
        except:
            exit("Error: mnist_handwritten_test.json not found")
        content = json.loads(content)
        try:
            X = []
            Y = []
            for x in content:
                X.append(x["image"][:])
                Y.append(x["label"])
        except:
            exit("Error: incorrect JSON format")
        self.X_test = np.array(X, float)
        self.Y_test = np.zeros((self.X_test.shape[0], 10), float)

        # preparing one-hot label Y test
        i = 0
        for y in Y:
            self.Y_test[i][y] = 1
            i += 1

    def get_X_train(self):
        print("X_train shape:", self.X_train.shape)
        return self.X_train
    def get_Y_train(self):
        print("Y_train shape:", self.Y_train.shape)
        return self.Y_train
    def get_X_test(self):
        print("X_test shape:", self.X_test.shape)
        return self.X_test
    def get_Y_test(self):
        print("Y_test shape:", self.Y_test.shape)
        return self.Y_test