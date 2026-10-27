# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the Iris dataset
dataset = pd.read_csv('iris.csv')

# Separate features and target
feature = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
X = dataset[feature]
y = dataset['target']

print(y)

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

# Apply feature scaling on the training and test sets
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Print the scaled training and test sets

print(X_train)
print(X_test)