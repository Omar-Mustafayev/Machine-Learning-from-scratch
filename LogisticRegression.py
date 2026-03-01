import numpy as np 
import pandas as pd
np.random.seed(218)

data = pd.read_csv(
'https://archive.ics.uci.edu/ml/'
'machine-learning-databases/iris/iris.data',
header=None, encoding='utf-8')

# select setosa and versicolor for simplicity
raw_df = data.iloc[0:100]
df = raw_df.iloc[np.random.permutation(len(raw_df))].reset_index(drop=True)
y = df.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 0, 1).reshape(-1, 1)
X = df.iloc[:, :4].values


X = np.hstack((np.ones((X.shape[0],1)), X)) #reserve x0 = 1 for bias 
m = X.shape[0]
n = X.shape[1]

weights = np.random.rand(n, 1) # initialize weights (and bias) to random values


def fit(X, y, learning_rate, epochs):
    global weights
    for iter in range(epochs):
        net_input = calculate_net_input(X, weights)
        output = sigmoid(net_input)
        weights += learning_rate*gradients(X, output, y)
        if (iter%10 == 0):
            print(f"{iter}-th epoch, Weights: {weights}")
    return weights
        

        
def calculate_net_input(X, thetas):
    return X @ thetas

# activation function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def gradients(X, predicted, target):
    return X.T @ (target-predicted)
    
print(fit(X, y, 0.1, 100))