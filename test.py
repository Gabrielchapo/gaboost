from Model.LinearRegression import LinearRegression
from Model.CrossValidation import CrossValidation
import numpy as np

# EXTRACT DATA FROM FILE
"""
matrix = np.loadtxt("text", delimiter=',', skiprows=1).T
X, Y = matrix[0], matrix[1]
"""
matrix = np.loadtxt("data/Car.csv", delimiter=';', skiprows=1)
Y = [x[-1] for x in matrix]
X = [x[0:-1] for x in matrix]


model = LinearRegression(normalize=True)

"""model.fit(X,Y)


print(model.predict([[8.860e+01, 1.688e+02, 6.410e+01, 4.880e+01, 2.548e+03, 1.300e+02,
       3.470e+00, 2.680e+00, 9.000e+00, 1.110e+02, 5.000e+03, 2.100e+01,
       2.700e+01]]))"""

tmp = CrossValidation(model, X, Y, 5)
print(tmp.accuracy())