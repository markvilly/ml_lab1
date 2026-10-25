import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


#dataset 
dataset = pd.read_csv('titanic.csv')

#data broken
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
x = dataset[features]
y = dataset.iloc[:, 1].values


#Sex is in column 1 and Embarked is in column 6 within features

ct  = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Sex', 'Embarked','Pclass'])], remainder='passthrough')
#NOte: only use Imputers when we have missing values-- OneHotEnocoder() fails on blanks.

x = np.array(ct.fit_transform(dataset)) 

le = LabelEncoder()
y = le.fit_transform(dataset['Survived'])

print(x)
print(y)