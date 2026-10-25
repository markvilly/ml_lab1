import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
np.set_printoptions(suppress=True)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


dataset = pd.read_csv('Data.csv')
# x = dataset.iloc[ :, :-1].values
# y = dataset.iloc[ :, -1].values


# imputer.fit(x[:, 1:3])

# #replace the data that's missing with imputed values
# x[:, 1:3] = imputer.transform(x[:, 1:3])

# print(x[:, 1:3])
# # print(y)

# dataset = pd.read_csv('pima-indians-diabetes.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(x[:, 1:6])

# x[:, 1:6] = imputer.transform(x[:, 1:6])

# print(x[:, 1:6])

#OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0] )], remainder='passthrough')
x = np.array(ct.fit_transform(x))
 
print(x)