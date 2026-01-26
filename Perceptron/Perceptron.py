import numpy as np # type: ignore
import pandas as pd # type: ignore
np.random.seed(218)

data = pd.read_csv(
'https://archive.ics.uci.edu/ml/'
'machine-learning-databases/iris/iris.data',
header=None, encoding='utf-8')
raw_df = data.iloc[0:100]
df = raw_df.iloc[np.random.permutation(len(raw_df))].reset_index(drop=True)

# select setosa and versicolor for simplicity
y = df.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 0, 1).reshape(-1, 1)
X = df.iloc[:, :4].values


X = np.hstack((np.ones((X.shape[0],1)), X)) #reserving x0 = 1 for bias 
m = X.shape[0]
n = X.shape[1]
initial_weights = np.random.rand(n, 1)
#print(X.shape, y.shape,  initial_weights.shape, (X @ initial_weights).shape)

def fit(X, y, learning_rate, epoch):
    weights = initial_weights.copy()
    print("Initial weights: ", weights)
    for iteration in range(1, epoch+1):
        update_count = 0
        for x_i, y_i, i in zip(X,y, range(1,m+1)):
            x_i = np.array([x_i])
            y_i = np.array([y_i])
            #print(x_i.shape,x_i, y_i.shape, y_i, i)
            prediction = predict(x_i, weights)
            adjustment = adjust(x_i, y_i, prediction)
            if (adjustment.any()):
                update_count += 1
            weights += learning_rate*adjustment
            if (i in range(10) or i % 10 == 0):
                print(f"{i}th sample in {iteration}th iteration. Weights: {weights}. Number of updates {update_count}")
    return weights

def predict(x_i, weights):
    activation_input = x_i @ weights
    return np.where(activation_input >= 0.0, 1, 0)

def adjust(x_i, true_label, predicted_label):
    return x_i.T * (true_label-predicted_label)

final_weights = fit(X, y, learning_rate=0.1, epoch=5)
