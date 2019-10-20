import numpy

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(A):
    expA = numpy.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def my_neural_network(X, tmp, nb_label):

	X = numpy.array(X, float)
	X = (X - 128) / 255

	numpy.random.seed(42)

	nb_training = X.shape[0]
	nb_features = X.shape[1]

	## fill label output with arrays
	Y = numpy.zeros((nb_training, nb_label), float)
	i = 0
	for y in tmp:
		Y[i][y] = 1
		i += 1

	hidden_nodes = 25

	theta_hidden = numpy.random.rand(nb_features, hidden_nodes)
	bias_hidden = numpy.random.randn(hidden_nodes)

	theta_output = numpy.random.rand(hidden_nodes, nb_label)
	bias_output = numpy.random.randn(nb_label)

	learning_rate = 0.00005
	
	for i in range(100):

		## FEEDFORWARD
		z_hidden = numpy.dot(X, theta_hidden) + bias_hidden
		a_hidden = sigmoid(z_hidden)

		z_output = numpy.dot(a_hidden, theta_output) + bias_output
		a_output = sigmoid(z_output)
		#a_output = z_output

		##BACK PROPAGATION
		sigma_output = a_output - Y
		cost_theta_output = numpy.dot(a_hidden.T, sigma_output)

		sigma_hidden = numpy.dot(sigma_output, theta_output.T) * sigmoid_der(z_hidden)
		cost_theta_hidden = numpy.dot(X.T, sigma_hidden)

		## UPDATE WEIGHTS
		theta_hidden += learning_rate * cost_theta_hidden
		bias_hidden += learning_rate * sigma_hidden.sum(axis = 0)

		theta_output += learning_rate * cost_theta_output
		bias_output += learning_rate * sigma_output.sum(axis = 0)

		print('Loss function value: ', numpy.sum(-Y * a_output))
	
	
	z_hidden = numpy.dot(X[42], theta_hidden)
	print(z_output)
	a_hidden = sigmoid(z_hidden)
	print(a_hidden)

	z_output = numpy.dot(a_hidden, theta_output)
	print(z_output)
	a_output = sigmoid(z_output)
	print(a_output)
