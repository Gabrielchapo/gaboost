import numpy as np
from GetMnistData import GetMnistData

def sigmoid_derv(s):
    return s * (1 - s)
def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples
def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss
def sigmoid(s):
    return 1/(1 + np.exp(-s))
def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

class MyNN:
    def __init__(self, x, y):
        self.input = x
        self.output = y
        neurons = 128       # neurons for hidden layers
        self.lr = 0.5       # user defined learning rate
        ip_dim = x.shape[1] # input layer size 64
        op_dim = y.shape[1] # output layer size 10
        self.w1 = np.random.randn(ip_dim, neurons) # weights
        self.b1 = np.zeros((1, neurons))           # biases
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
    def feedforward(self):
        z1 = np.dot(self.input, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)
    def backprop(self):
        loss = error(self.a3, self.output)
        print('Error :', loss)
        a3_delta = cross_entropy(self.a3, self.output) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1
        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.input.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)
    def predict(self, data):
        self.input = data
        self.feedforward()
        return self.a3.tolist()

if __name__ == "__main__":
        	
    data = GetMnistData()
    model = MyNN(data.get_X_train(), data.get_Y_train())
    for _ in range(50):
        model.feedforward()
        model.backprop()