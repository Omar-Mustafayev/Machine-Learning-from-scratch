import numpy as np # type: ignore
from sklearn.datasets import fetch_california_housing # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
np.set_printoptions(suppress=True)


housing = fetch_california_housing()
X = housing.data  
y_train = housing.target.reshape(-1, 1)  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))


n = X_train.shape[0]
m = X_train.shape[1]
theta_init = np.random.rand(m, 1)

def compute_cost(X, y, theta):
    errors = (X @ theta - y)**2
    cost = np.sum(errors) / (2 * n)
    return cost

def compute_gradient(X, y, theta):
    err = X @ theta - y
    gradient = (X.T @ err) / n
    return gradient


def gradient_descent(X, y, learning_rate, iterations):
    thetas = theta_init.copy()
    for iter in range(iterations + 1):
        if iter % 10 == 0:
            cost = compute_cost(X, y, thetas)
            print(f"Iteration: {iter}  Thetas: {thetas.ravel()}  Cost: {cost}")
        gradient = compute_gradient(X, y, thetas)
        thetas-= learning_rate * gradient
    return thetas


final_thetas = gradient_descent(X_train, y_train, learning_rate=0.02, iterations=15000)



lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y_train)
print("Intercept:", lin_reg.intercept_)
print("Coefficients:", lin_reg.coef_)
print("Our intercept:", final_thetas[0])
print("Our coefficients:", final_thetas[1:])

