from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
from GetMnistData import GetMnistData
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

data = GetMnistData()

model = Sequential()
model.add(Dense(32, activation='sigmoid', input_dim=784))
model.add(Dense(32, activation='sigmoid', input_dim=784))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(data.get_X_train(), data.get_Y_train(), epochs=20, batch_size=32, verbose=2)

prediction = model.predict(data.get_X_test()[0:2], verbose=1)

print(np.argmax(prediction, axis=1))