import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
                X.append(x["image"])
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
                X.append(x["image"])
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
        return self.X_train
    def get_Y_train(self):
        return self.Y_train
    def get_X_test(self):
        return self.X_test
    def get_Y_test(self):
        return self.Y_test

def cross_entropy(Y_pred, Y_real):
    diff = Y_pred - Y_real
    return diff / Y_real.shape[0]

def error(Y_pred, Y_real):
    logged = - np.log(Y_pred[np.arange(Y_real.shape[0]), Y_real.argmax(axis=1)])
    return np.sum(logged) / Y_real.shape[0]

def sigmoid_derv(s):
    return s * (1 - s)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.nb_layers = 0

    def feedforward(self):
        tmp = self.input
        for layer in self.layers:
            name = layer["name"]
            z = np.dot(tmp, layer["theta"]) + layer["bias"]
            if layer["activation"] == "sigmoid":
                self.activated[name] = sigmoid(z)
            elif layer["activation"] == "softmax":
                self.activated[name] = softmax(z)
            else:
                print("Error: Unknown or invalid activation function")
                return
            tmp = self.activated[name]

    def backprop(self):

        activated = None
        for attribut in reversed(self.layers):
            name = attribut["name"]
            if activated is None:
                self.activated_delta[name] = cross_entropy(self.activated[name], self.output)
            else:
                self.activated_delta[name] = np.dot(activated, tmp) * sigmoid_derv(self.activated[name])
            
            tmp = attribut["theta"].T
            activated = self.activated_delta[name]
        i = self.nb_layers - 1
        while i >= 0:
            name = self.layers[i]["name"]
            if i != 0:
                self.layers[i]["theta"] -= self.lr * np.dot(self.activated[self.layers[i - 1]["name"]].T, self.activated_delta[name])
            else:
                self.layers[i]["theta"] -= self.lr * np.dot(self.input.T, self.activated_delta[name])
            if i == self.nb_layers - 1:
                self.layers[i]["bias"] -= self.lr * np.sum(self.activated_delta[name], axis=0, keepdims=True)
            else:
                self.layers[i]["bias"] -= self.lr * np.sum(self.activated_delta[name], axis=0)
            i -= 1

    def predict(self, data):
        self.input = data
        self.feedforward()
        name = self.layers[self.nb_layers - 1]["name"]
        return self.activated[name].tolist()

    def add_layer(self, size, activation, input_dim=None):

        layer = {}

        if bool(self.layers) == False:
            layer["name"] = "layer_0"
            if input_dim is None:
                print("First layer need an input dimension")
                return
        else:
            i = self.nb_layers
            layer["name"] = "layer_" + str(i)
            input_dim = self.layers[i - 1]['theta'].shape[1]

        layer['activation'] = activation
        layer['theta'] = np.random.randn(input_dim, size)
        layer['bias'] = np.zeros((1, size))
        self.layers.append(layer)
        self.nb_layers += 1

    def summary(self):
        for attribut in self.layers:
            print("Layer:", attribut["name"], "| Dimensions:", attribut["theta"].shape, "| Activation:", attribut["activation"])

    def compile(self, lr, loss):
        self.lr = lr
        self.loss = loss
        self.activated = {}
        self.activated_delta = {}

    def fit(self, X, Y, epoch, verbose=0):

        self.input = X
        self.output = Y
        self.err = []

        if epoch <= 0:
            print("Invalid number of epochs")
            return
        for i in range(epoch):
            self.feedforward()
            self.backprop()
            err = error(self.activated[self.layers[self.nb_layers - 1]["name"]], self.output)
            self.err.append(err)
            if verbose == 1:
                print("Epoch:", i + 1, "/", epoch, "=== Loss:", err)
        fig, ax = plt.subplots()
        ax.plot(self.err)
        ax.set(xlabel='epochs', ylabel='loss')
        plt.show()

    def load(self, path):
        try:
            with open(path, 'r') as file:
                content = file.read()
                file.close()
        except:
            print("Error: json file not found")
        content = json.loads(content)
        for attribut in content:
            layer = {}
            layer["name"] = attribut["name"]
            layer["activation"] = attribut["activation"]
            layer["theta"] = np.array(attribut["theta"])
            layer["bias"] = np.array(attribut["bias"])
            self.layers.append(layer)
            self.nb_layers += 1

    def save(self, path):
        i = self.nb_layers - 1
        while i >= 0:
            self.layers[i]["theta"] = self.layers[i]["theta"].tolist()
            self.layers[i]["bias"] = self.layers[i]["bias"].tolist()
            i -= 1
        content = json.dumps(self.layers)
        with open(path, "w") as file:
            file.write(content)
            file.close()
