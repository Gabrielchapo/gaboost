from GetMnistData import GetMnistData
from tmp import MyNN
import numpy as np


data = GetMnistData()

X = data.get_X_train()
Y = data.get_Y_train()

X_test = data.get_X_test()
Y_test = data.get_Y_test()

model = MyNN(X, Y)

for i in range(100):
    model.feedforward()
    model.backprop()

prediction = model.predict(X_test)

prediction = np.argmax(prediction, axis=1)
real = np.argmax(Y_test, axis=1)

count = 0

for i in range(len(prediction)):
    print("predicted:",prediction[i], ', real:', real[i])
    if prediction[i] == real[i]:
        count += 1

print("Accuracy: ", count / len(prediction))