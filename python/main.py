from GetMnistData import GetMnistData
from MyNN import MyNN

if __name__ == "__main__":
        	
    data = GetMnistData()
    model = MyNN(data.get_X_train(), data.get_Y_train())
    for _ in range(50):
        model.feedforward()
        model.backprop()