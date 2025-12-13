import numpy as np # type: ignore
import pandas as pd # type: ignore
np.random.seed(217)

df = pd.read_csv(
'https://archive.ics.uci.edu/ml/'
'machine-learning-databases/iris/iris.data',
header=None, encoding='utf-8')


# select setosa and versicolor for simplicity
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1).reshape(-1, 1)
X = df.iloc[0:100, 0:4].values

X = np.hstack((np.ones((X.shape[0],1)), X)) #reserving x0 = 1 for bias 
m = X.shape[0]
n = X.shape[1]
initial_weights = np.random.rand(n, 1)
print(X.shape, y.shape,  initial_weights.shape, (X @ initial_weights).shape)

def fit(X, y, learning_rate, epoch):
    weights = initial_weights
    for iteration in range(epoch):
        for x_i, y_i, i in zip(X,y, range(100)):
            x_i = np.array([x_i])
            y_i = np.array([y_i])
            #print(x_i.shape,x_i, y_i.shape, y_i, i)
            prediction = predict(x_i, weights)
            adjustment = adjust(x_i, y_i, prediction)
            weights += learning_rate*adjustment
            if (i%99 == 0):
                print(f"{i}th sample in {iteration}th iteration. Weights: {weights}")
    return weights
def predict(x_i, weights):
    activation_input = x_i @ weights
    return np.where(activation_input >= 0.0, 1, 0)

def adjust(x_i, true_label, predicted_label):
    return x_i.T * (true_label-predicted_label)

final_weights = fit(X, y, learning_rate=0.2, epoch=100)
