#!/usr/bin/env python
# coding: utf-8

# In[2]:


#1 Take any 2 features from your project having has real numeric values. Make a scatter plot of the data and observe the pattern.


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


# In[3]:


#2 Create a linear regression model on this data. Consider using one feature as independent variable while the other as dependent variable (you may also round this number to integer). After the model is created, calculate the mean square error by predicting the values from the model. Refer site: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html


# In[4]:


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


# In[4]:


#4 Using the training set available for your project, train a logistic regression classifier. Use this classifier to evaluate your test set accuracy. Study the various parameters associated with logistic regression model and the role they play in the model training.[NOTE]: For Logistic regression, use bi-class classification problem. If your data has multiple classes, take any two classes. If your data has value prediction problem (Ex: stock price or car value prediction), convert it into a bi-class classification problem and fit a logistic regressor.


# In[11]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the training and testing datasets
training_data = pd.read_excel('training (2) (1).xlsx')
test_data = pd.read_excel('testing (2) (1).xlsx')

# Clean the training data by filling NaN values with an empty string
training_data['input'].fillna('', inplace=True)

# Clean the test data by filling NaN values with an empty string
test_data['Equation'].fillna('', inplace=True)

# Assuming 'input', 'output', and 'classification' are columns in your training data
X_train = training_data[['input', 'output']]
y_train = training_data['Classification']

# Assuming 'Equation' and 'output' are columns in your test data
X_test = test_data[['Equation', 'output']]
y_test = test_data['Classification']

# Initialize a CountVectorizer to convert text to numeric features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train['input'])  # Convert the 'input' column
X_test = vectorizer.transform(X_test['Equation'])  # Convert the 'Equation' column

# Initialize and train the logistic regression model
model = LogisticRegression(solver='liblinear', penalty='l2', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on the test data: {accuracy}")


# In[14]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the training and testing datasets
training_data = pd.read_excel('training (2) (1).xlsx')
test_data = pd.read_excel('testing (2) (1).xlsx')

# Clean the training data by filling NaN values with an empty string
training_data['input'].fillna('', inplace=True)

# Clean the test data by filling NaN values with an empty string
test_data['Equation'].fillna('', inplace=True)

# Assuming 'input', 'output', and 'classification' are columns in your training data
X_train = training_data[['input', 'output']]
y_train = training_data['Classification']

# Assuming 'Equation' and 'output' are columns in your test data
X_test = test_data[['Equation', 'output']]
y_test = test_data['Classification']

# Initialize a CountVectorizer to convert text to numeric features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train['input'])  # Convert the 'input' column
X_test = vectorizer.transform(X_test['Equation'])  # Convert the 'Equation' column

# Use PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train.toarray())
X_test_pca = pca.transform(X_test.toarray())

# Initialize and train the logistic regression model
model = LogisticRegression(solver='liblinear', penalty='l2', C=1.0, max_iter=1000)
model.fit(X_train_pca, y_train)

# Plot decision boundary
h = .02
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

# Create scatter plots of the data points
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


# In[5]:


#5 Use a Regression Tree and k-NN Regressor for value prediction. Use the data employed for A1


# In[4]:


print(X_train.shape)
print(y_train.shape)


# In[10]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the training data
train_data = pd.read_excel('training (2) (1).xlsx')

# Fill NaN values with an empty string
train_data['input'].fillna('', inplace=True)

# Assuming 'input_' and 'output' are columns in your training data
X = train_data['input']
y = train_data['Classification']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and fit the Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train_tfidf, y_train)

# Predictions
y_pred = decision_tree_classifier.predict(X_test_tfidf)

# Calculate and print accuracy
accuracy = (y_pred == y_test).mean()

print("Decision Tree Accuracy:", accuracy)


# In[12]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the training and test data
train_data = pd.read_excel('training (2) (1).xlsx')
test_data = pd.read_excel('testing (2) (1).xlsx')

# Ensure that the column names are consistent
train_data = train_data.rename(columns={"input": "input_", "output": "output"})
test_data = test_data.rename(columns={"Equation": "input_", "output": "output"})

# Define a function to preprocess the 'input_' column
def preprocess_input(input_data):
    # Convert equations to strings
    input_data = input_data.astype(str)
    # Encode the input data using LabelEncoder
    label_encoder = LabelEncoder()
    input_encoded = label_encoder.fit_transform(input_data)
    return input_encoded

# Preprocess the input data
X_train = preprocess_input(train_data['input_'])
X_test = preprocess_input(test_data['input_'])

# Extract 'output' and 'Classification' columns
y_train = train_data['Classification']
y_test = test_data['Classification']

# Train a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train.reshape(-1, 1), y_train)

# Train a k-NN classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train.reshape(-1, 1), y_train)

# Predictions
decision_tree_predictions = decision_tree_classifier.predict(X_test.reshape(-1, 1))
knn_predictions = knn_classifier.predict(X_test.reshape(-1, 1))

# Calculate and print accuracy
decision_tree_accuracy = (decision_tree_predictions == y_test).mean()
knn_accuracy = (knn_predictions == y_test).mean()

print("Decision Tree Accuracy:", decision_tree_accuracy)
print("k-NN Accuracy:", knn_accuracy)


# In[17]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the training and test data
train_data = pd.read_excel('training (2) (1).xlsx')
test_data = pd.read_excel('testing (2) (1).xlsx')

# Ensure that the column names are consistent
train_data = train_data.rename(columns={"input": "input_", "output": "output"})
test_data = test_data.rename(columns={"Equation": "input_", "output": "output"})

# Remove rows with missing 'input_' values
train_data = train_data.dropna(subset=['input_'])
test_data = test_data.dropna(subset=['input_'])

# Extract 'input_' and 'output' columns
X_train = train_data['input_']
y_train = train_data['Classification']
X_test = test_data['input_']
y_test = test_data['Classification']

# Initialize a CountVectorizer to convert text data to numeric features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree_classifier, filled=True, feature_names=vectorizer.get_feature_names_out().tolist(), class_names=["0", "1"])
plt.title("Decision Tree Visualization")
plt.show()


# In[ ]:




