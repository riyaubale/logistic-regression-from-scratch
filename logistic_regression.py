"""
Logistic Regression vs Least Squares

- Implements logistic regression using gradient descent
- Compares with squared error classifier
- Evaluates robustness to outliers
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


# ----------------------------
# Load Data
# ----------------------------
with open('classifier_data.pkl', 'rb') as f:
    X, y = pickle.load(f)

plt.scatter(X[:,0], X[:,1], c=y[:,0])
plt.title("Training Data")
plt.show()


# ----------------------------
# Logistic Regression
# ----------------------------
def gradient_descent(X, y, lr, lam, w, iters):
    for _ in range(iters):
        margins = y * (X @ w)
        grad = (-y / (1 + np.exp(margins))) @ X + 2 * lam * w
        w = w - lr * grad
    return w


X_aug = np.hstack([X, np.ones((X.shape[0],1))])
w = gradient_descent(X_aug, y, 0.001, 0.1, np.random.randn(3), 1000)

y_pred = np.sign(X_aug @ w)
error = np.mean(y_pred != y)

print("Logistic Error:", error)

plt.scatter(X[:,0], X[:,1], c=y_pred)
plt.title("Logistic Regression Predictions")
plt.show()


# ----------------------------
# Least Squares
# ----------------------------
def squared_error(X, y, lam):
    return np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y


w_ls = squared_error(X_aug, y, 1)
y_pred_ls = np.sign(X_aug @ w_ls)
error_ls = np.mean(y_pred_ls != y)

print("Least Squares Error:", error_ls)


# ----------------------------
# Outlier Experiment
# ----------------------------
n_new = 1000
X_out = np.vstack([X, np.hstack([10*np.ones((n_new,1)), np.zeros((n_new,1))])])
y_out = np.vstack([y, -1*np.ones((n_new,1))])

X_aug_out = np.hstack([X_out, np.ones((X_out.shape[0],1))])

w_log = gradient_descent(X_aug_out, y_out, 0.001, 1, np.random.randn(3), 1000)
w_ls_out = squared_error(X_aug_out, y_out, 1)

print("Logistic vs LS robustness test complete.")