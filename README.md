# Machine Learning from scratch

The objective is to implement a Neural Network for classification problem.

In Python => Folder Python

In C      => folder C (work in progress)

## Getting Started

You can download the MNIST DATASET (Train and test separated) in json encoded and put it in /data file, and use the GetMnistData class to load it.
Follow this link to download it : https://github.com/lorenmh/mnist_handwritten_json 

Feel free to use your own dataset.

### Usage

First, import the MyNN class
```
from MyNN import MyNN
```
Then, initialize the class
```
model = MyNN()
```
Then add layers with the density, the input dimensions for the first layer, and the activation function
```
model.add_layer(64, input_dim=X.shape[1], activation='sigmoid')
model.add_layer(64, activation='sigmoid')
model.add_layer(10, activation='softmax')
```
You can check your model with
```
model.summary()
```
Compile your model with the learning rate and loss function
```
model.compile(1, "cross_entropy")
```
Now your model is ready to be fitted, give the input and the output data, the number of epoch and in option if you want information on each epochs
```
model.fit(X, Y, epoch=2000, verbose=1)
```
You can now predict with your model
```
prediction = model.predict(X_test)
```

Other way to create your model:
Use model.load() and model.save() with the path as a paremeter to get and use your model in a json file
```
model.load("here.json")
model.save("here.json")
```

Don't hesitate to run the project with the main.py file to get an example
```
python3 main.py
```

## Author

* **Gabriel Drai** - *Initial work* 

## License

https://github.com/Gabrielchapo/machine_learning/blob/master/LICENSE
