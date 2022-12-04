import pandas as pd
import numpy as np
import matplotlib.pyplot as plotter


def getdata():
    data = np.array(pd.read_csv("linear-regression.csv"))
    income = data[:, 0]
    expenditure = data[:, 1]
    return income, expenditure


def train(X, Y):
    # Function to train the model
    # We are using Ordinary Least Squares function for this
    X_mean = np.mean(X)
    y_mean = np.mean(Y)

    # Squared sum of Xy and XX
    sum_xy = 0
    sum_xx = 0

    # Calculate the numerator and denominator
    for i in range(len(X)):
        x_diff = X[i] - X_mean
        y_diff = Y[i] - y_mean

        sum_xy = sum_xy + (x_diff * y_diff)
        sum_xx = sum_xx + pow(x_diff, 2)

    # Calculate the coefficients
    b1 = sum_xy/sum_xx
    b0 = y_mean - (X_mean * b1)

    # Return the parameters
    return b0, b1


if __name__ == "__main__":
    income, expenditure = getdata()
    a, b = train(income, expenditure)
    print('The equation for linear regression is y =', a, '+', b, '* x')
    predicted_values = a + b*income
    plotter.scatter(income, expenditure)
    plotter.plot(income, predicted_values, color="y")
    plotter.show()
