# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Loading the data
df = pd.read_csv('data.csv')

# Setting up our X and Y
X = df[['rating']]
Y = df['salary']

# Splitting the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Fitting a linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predicting salary using the fitted model
Y_pred = model.predict(X_test)

# Calculating the MAPE
mape = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100

# Printing the coefficients and MAPE
print(f"{model.intercept_:.5f} {model.coef_[0]:.5f} {mape:.5f}")

# Scatterplot
plt.scatter(X, Y, color = 'blue')
plt.plot(X_test, Y_pred, color = 'red')
plt.title('Linear Regression')
plt.xlabel('Rating')
plt.ylabel('Salary')
plt.show()
