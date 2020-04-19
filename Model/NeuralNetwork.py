import numpy as np
import json
import C_module

class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.nb_layers = 0

    def add_layer(self, size, activation=None, input_dim=None):

        layer = {}
        np.random.seed(42)
        if self.nb_layers == 0:
            if input_dim == None:
                exit("Error: first layer need an input_dim.")
        else:
            input_dim = self.layers[-1]["weights"].shape[1]

        layer["name"] = "layer_"+str(self.nb_layers)
        layer['activation'] = activation
        layer['weights'] = np.random.randn(input_dim, size)
        layer['bias'] = np.zeros((1, size))
        self.layers.append(layer)
        self.nb_layers += 1

    def summary(self):
        for attribut in self.layers:
            print("Layer:", attribut["name"], "| Dimensions:", attribut["weights"].shape, "| Activation:", attribut["activation"])

    def compile(self, lr, loss):
        self.lr = lr
        self.loss = loss

    def fit(self, X, Y, epoch, normalize=False):
        self.sigma = [np.amax(x) - np.amin(x) if np.amax(x) - np.amin(x) != 0 else 1 for x in zip(*X)]
        self.mean = [sum(x) / len(X) for x in zip(*X)]
        X = (X - self.mean) / self.sigma
        X = [list(x) for x in X]
        Y = [list(y) for y in Y]
        b = []
        W = []
        for layer in self.layers:
            W.append(layer["weights"].tolist())
            b.append(layer['bias'].tolist())
        tmp = C_module.neural_network_fit(X,Y,W,b,epoch)
        W = tmp[:self.nb_layers]
        b = tmp[self.nb_layers:]
        for i,x in enumerate(W):
            self.layers[i]["weights"] = x
        for i,x in enumerate(b):
            self.layers[i]["bias"] = x
            
        
    def predict(self, X):
        self.sigma = [np.amax(x) - np.amin(x) if np.amax(x) - np.amin(x) != 0 else 1 for x in zip(*X)]
        self.mean = [sum(x) / len(X) for x in zip(*X)]
        X = (X - self.mean) / self.sigma
        X = [list(x) for x in X]
        b = []
        W = []
        for layer in self.layers:
            W.append(layer["weights"])
            b.append(layer['bias'])
        return C_module.neural_network_predict(X,W,b)


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
            layer["weights"] = np.array(attribut["weights"])
            self.layers.append(layer)
            self.nb_layers += 1

    def save(self, path):
        for index,layer in enumerate(self.layers):
            self.layers[index]["weights"] = layer["weights"].tolist()
        content = json.dumps(self.layers)
        with open(path, "w") as file:
            file.write(content)
            file.close()
