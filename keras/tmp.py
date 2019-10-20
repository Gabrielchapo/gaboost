import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import sys
import json


#########  IMPORTING TRAINING DATA ############
try:
	with open(sys.argv[1], 'r') as file:
		content = file.read()
		file.close()
except:
	print("Usage : ./machine_learning <src_file.JSON>")
content = json.loads(content)

try:
	X = []
	tmp = []
	for x in content:
		X.append(x["image"])
		tmp.append(x["label"])
except:
	print("Error: incorrect JSON format")

X = np.array(X, float)
#X = (X - 128) / 255

Y = np.zeros((X.shape[0], 10), float)
i = 0
for y in tmp:
	Y[i][y] = 1
	i += 1

########  IMPORTING TEST DATA ############

try:
	with open(sys.argv[2], 'r') as file:
		content = file.read()
		file.close()
except:
	print("Usage : ./machine_learning <src_train.JSON> <src_test.JSON>")
content = json.loads(content)

try:
	X_test = []
	tmp_test = []
	for x in content:
		X_test.append(x["image"])
		tmp_test.append(x["label"])
except:
	print("Error: incorrect JSON format")

X_test = np.array(X_test, float)

Y_test = np.zeros((X_test.shape[0], 10), float)
i = 0
for y in tmp_test:
	Y_test[i][y] = 1
	i += 1

########  MODEL CREATION ############
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


########  MODEL TRAINING ############
model.fit(X, Y, epochs=10, batch_size=32, verbose=2)

########  MODEL TEST ############
prediction = model.predict(X_test[:2], verbose=1)

print(np.argmax(prediction, axis=1))