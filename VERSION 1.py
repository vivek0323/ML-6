#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the testing and training datasets
test_data = pd.read_excel("testing (2) (1).xlsx")
train_data = pd.read_excel("training (2) (1).xlsx")

# Extract the 'Output' and 'Classification' columns
test_output = test_data['output']
test_classification = test_data['Classification']

train_output = train_data['output']
train_classification = train_data['Classification']

# Create scatter plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(test_output, test_classification, c='red', label='Test Data')
plt.xlabel('output')
plt.ylabel('Classification')
plt.title('Scatter Plot for Test Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(train_output, train_classification, c='blue', label='Train Data')
plt.xlabel('output')
plt.ylabel('Classification')
plt.title('Scatter Plot for Train Data')
plt.legend()

plt.tight_layout()
plt.show()


# In[2]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the training and testing datasets
train_data = pd.read_excel("training (2) (1).xlsx")
test_data = pd.read_excel("testing (2) (1).xlsx")

# Define independent (X) and dependent (y) variables for training and testing datasets
X_train = train_data[['output']].values  # Independent variable for training
y_train = train_data['Classification'].values  # Dependent variable for training

X_test = test_data[['output']].values  # Independent variable for testing
y_test = test_data['Classification'].values  # Dependent variable for testing

# Create and fit the linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficients and calculate the mean squared error for training data
coefficients = model.coef_
mse_train = mean_squared_error(y_train, model.predict(X_train))

# Calculate the mean squared error for testing data
mse_test = mean_squared_error(y_test, model.predict(X_test))

print("Coefficients:", coefficients)
print("Mean squared error on training data:", mse_train)
print("Mean squared error on testing data:", mse_test)

# Plot the scatter plot with the regression line for both training and testing data
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression Line')
plt.xlabel('output')
plt.ylabel('Classification')
plt.title('Linear Regression on Training Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Testing Data')
plt.plot(X_test, model.predict(X_test), color='red', linewidth=2, label='Regression Line')
plt.xlabel('output')
plt.ylabel('Classification')
plt.title('Linear Regression on Testing Data')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




