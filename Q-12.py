#Volkan Işık  15 April 2022

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def safelog(x):
    return(np.log(x+ 1e-100))

data = np.genfromtxt("dataset.csv", skip_header = 1 ,usecols= [0,1,2,3],
                     delimiter = ",", filling_values= "999999")
data_2 = np.genfromtxt("dataset.csv", skip_header = 1 ,usecols= [4],
                     delimiter = ",",  dtype= [("U5")])

N = data.shape[0]
D = 4
N_train = math.floor(N*0.75)
N_test = N- N_train


Y = np.asarray(data_2["f0"])
Y = np.reshape(Y,[len(Y),1])


X = np.zeros((N_train,D))
X_test = np.zeros((N_test,D))
Y_train = np.zeros((N_train,1)).astype(str)
Y_test = np.zeros((N_test,1)).astype(str)

# As my data can be similar in some areas i divide the data randomly to test and train set
iteration = 0
for i in np.random.choice(range(N), N, replace = False):
    if iteration < N_test:
        X_test[iteration, :] = data[i, :]
        Y_test[iteration, :] = Y[i, :]
    else:
        X[iteration-N_test, :] = data[i, :]
        Y_train[iteration - N_test, :] = Y[i, :]
    iteration += 1
Y = Y_train

# Here i turn string data type to integers
for i in range(Y.shape[0]):
    if Y[i,0] == "False":
        Y[i] = 0
    elif Y[i,0] == "True":
        Y[i] = 1
Y = Y.astype(int)
for i in range(Y_test.shape[0]):
    if Y_test[i,0] == "False":
        Y_test[i] = 0
    elif Y_test[i,0] == "True":
        Y_test[i] = 1
Y_test = Y_test.astype(int)

#Here in below for loop i replaced the missing values with the means of the column
for c in range(4):
    mean_1 = np.mean(X[:,c] != 999999)
    mean_2 = np.mean(X_test[:, c] != 999999)
    for i in range(N_train):
        if (X[i,c] == 999999 ):
            X[i,c] = mean_2
    for i in range(N_test):
        if (X[i,c] == 999999 ):
            X[i,c] = mean_2

# I used sigmoid because we are working on a binary classification problem
# I used a hidden layer which has 40 nodes
# I trained my data with changing weights by adding their gradients
# I used log likelihood as a score function

def sigmoid(a):
    return(1 / (1 + np.exp(-a)))

# These are my initial parameters which can be changed
eta = 0.1
epsilon = 0.001
H = 40
max_iteration = 500

# W is weights of input to hidden nodes
W = np.random.uniform(low = -0.01, high = 0.01, size = (D + 1, H))
# V is weights of hidden nodes to output
V = np.random.uniform(low = -0.01, high = 0.01, size = H + 1)

# Z is hidden layer
Z = sigmoid(np.matmul(np.hstack((np.ones((N_train,  1)), X)), W))
Y_predicted = sigmoid(np.matmul(np.hstack((np.ones((N_train,  1)), Z)), V))
Y_predicted = Y_predicted.reshape(Y_predicted.shape[0],1)


error = -np.sum(Y * safelog(Y_predicted) + (1 - Y) * safelog(1 - Y_predicted))



errors = np.array([error])
iteration = 1
while True:
    W_old = W
    V_old = V
    # I am randomly selecting a data point to change my weights
    for i in np.random.choice(range(N_train), N_train, replace = False):
        Z[i, :] = sigmoid(np.matmul(np.hstack((1, X[i, :])), W))
        Y_predicted = sigmoid(np.matmul (np.hstack((1, Z[i, :])), V))

        # deltas are gradients of V and W times a constant eta
        delta_V = eta * (Y[i] - Y_predicted) * np.hstack((1, Z[i,:]))
        delta_W = eta * (Y[i] - Y_predicted) * np.matmul( np.hstack((1, X[i,:])).reshape((D+1,1)),
                                                (V[1:] * Z[i, :] * (1 - Z[i, :])).reshape((1,H)))

        V = V + delta_V
        W = W + delta_W

    # After I changed my V and W with all inputs I calculate my results for that iteration
    Z = sigmoid(np.matmul(np.hstack((np.ones((N_train, 1)), X)), W))
    Y_predicted = sigmoid(np.matmul(np.hstack((np.ones((N_train, 1)), Z)), V))
    Y_predicted = Y_predicted.reshape(Y_predicted.shape[0], 1)
    error = -np.sum(Y * safelog(Y_predicted) + (1 - Y) * safelog(1 - Y_predicted))
    errors = np.append(errors, error)

    # I am checking if my code needs to stop
    if np.sqrt(np.sum((V - V_old) ** 2) + np.sum((W - W_old) ** 2)) < epsilon or iteration >= max_iteration:
        break

    iteration = iteration + 1

# This is a plot of error by iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 2), errors, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# I am converting the floating points to integers and printing a confusion matrix
Y_predicted = 1 * (Y_predicted > 0.5)
confusion_matrix= pd.crosstab(Y_predicted.flatten(),Y.flatten(),
                                   rownames = ["y_pred"], colnames = ["y_truth"])
print(confusion_matrix)
print("\n\n")

# I am calculating the predicted values of test set
Z = sigmoid(np.matmul(np.hstack((np.ones((N_test,  1)), X_test)), W))
Y_predicted_test = sigmoid(np.matmul(np.hstack((np.ones((N_test,  1)), Z)), V))
Y_predicted_test = Y_predicted_test.reshape(Y_predicted_test.shape[0],1)
Y_predicted_test = 1 * (Y_predicted_test > 0.5)

confusion_matrix_test= pd.crosstab(Y_predicted_test.flatten(),Y_test.flatten(),
                                   rownames = ["y_pred_test"], colnames = ["y_truth_test"])
print(confusion_matrix_test)













