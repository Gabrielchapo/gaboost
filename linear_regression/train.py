import numpy as np
import matplotlib.pyplot as plt

# EXTRACT DATA FROM FILE
matrix = np.loadtxt("text", delimiter=',', skiprows=1).T
X, Y = matrix[0], matrix[1]

# HYPERPARAMETERS
epochs = 3000
alpha = 0.05

# USEFUL DATA
m = X.shape[0]

# MEAN NORMALIZATION
mu = sum(X) / m
sigma = np.amax(X) - np.amin(X)
X = (X - mu) / sigma

# INCLUDE THE BIAS
X = X.reshape((m,1))
X = np.insert(X, 0, 1, axis=1)

# THETA INITIALIZATION
theta = [1,2]

for i in range(epochs):
    hypothesis = np.dot(X, theta)
    
    diff = (hypothesis - Y)
    
    tmp = (alpha / m) * np.sum(diff * X.T, axis=1) 

    theta -= tmp

    # COST FUNCTION
    print("cost:", (1/m) * sum(diff * diff))

# NORMAL EQUATION
step1 = np.dot(X.T, X)
step2 = np.linalg.pinv(step1)
step3 = np.dot(step2, X.T)
thetas = np.dot(step3, Y)

# PLOT IT
plt.title('ft_linear_regression')

y = np.dot(X, theta)
plt.plot(matrix[0], y, '-r', label='Gradient Descent')

y = np.dot(X, thetas)
plt.plot(matrix[0], y, '-b', label='Normal Equation')

plt.plot(matrix[0], Y, 'gs')
plt.xlabel('km', color='#1C2833')
plt.ylabel('price', color='#1C2833')
plt.legend(loc='upper right')
plt.show()

# SAVE WEIGHTS
f = open("weights", "w")
f.write(str(theta[0]) + ',' + str(theta[1]) + ',' + str(mu) + ',' + str(sigma))
f.close()
