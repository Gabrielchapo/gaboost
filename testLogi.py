from Model.LogisticRegression import LogisticRegression
from Model.CrossValidation import CrossValidation
import pandas as pd

df = pd.read_csv("data/titanic.csv")
Y = df.Survived.tolist()
Y = [[1,0] if x == 1 else [0,1] for x in Y]

df = df[["Pclass", "SibSp", "Parch", "Fare", "Sex"]]
df = pd.get_dummies(df)
X = df.values.tolist()

model = LogisticRegression(normalize=True)
"""
X = [[1,1,1,0,0,0],
    [1,0,1,0,0,0],
    [1,1,1,0,0,0],
    [0,0,1,1,1,0],
    [0,0,1,1,0,0],
    [0,0,1,1,1,0]]
Y = [[1, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 1]]"""
"""
model.fit(X,Y)
print(model.get_weights())
print(model.predict([3.0, 1.0, 0.0, 7.25]))
print(Y[0])"""

tmp = CrossValidation(model, X, Y, 5)

print(tmp.accuracy())