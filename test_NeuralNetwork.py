from Model.NeuralNetwork import NeuralNetwork
from Model.GetMnistData import GetMnistData
import numpy as np


data = GetMnistData('data/mnist_handwritten_train.json', 'data/mnist_handwritten_test.json')

X = data.get_X_train()
Y = data.get_Y_train()

X_test = data.get_X_test()
Y_test = data.get_Y_test()

model = NeuralNetwork()


model.add_layer(30, input_dim=X.shape[1], activation='sigmoid')
model.add_layer(30, activation='sigmoid')
model.add_layer(10, activation='softmax')

#model.load("weights.json")

model.summary()

model.compile(0.3, "cross_entropy")

model.fit(X, Y, epoch=3000, normalize=True)

prediction = model.predict(X_test)
prediction = np.argmax(prediction, axis=1)
real = np.argmax(Y_test, axis=1)
count = 0

for i in range(len(prediction)):
    #print("predicted:",prediction[i], ', real:', real[i])
    if prediction[i] == real[i]:
        count += 1

print("Accuracy: ", count / len(prediction))


#model.save("weights.json")
