import numpy as np
import sys

matrix = np.loadtxt("weights", delimiter=',')

try:
    km = int(sys.argv[1])
except:
    print("Parameter must be an integer")
    sys.exit()

km = (km - matrix[2]) / matrix[3]

print("Prediction:", matrix[1] * km + matrix[0])
