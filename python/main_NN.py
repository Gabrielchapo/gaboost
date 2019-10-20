import sys
import json
from my_neural_network import my_neural_network

try:
	with open(sys.argv[1], 'r') as file:
		content = file.read()
		file.close()
except:
	print("Usage : ./machine_learning <src_file.JSON> <dest_file.JSON>")
content = json.loads(content)
content = content[0:6000]

try:
	X = []
	Y = []
	for x in content:
		x['image'].insert(0, '1')
		X.append(x["image"])
		Y.append(x["label"])
except:
	print("Error: incorrect JSON format")

my_neural_network(X, Y, 10)