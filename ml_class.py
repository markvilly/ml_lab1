import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


dataset = pd.read_csv('Data.csv')
x = dataset.iloc[ :, :-1].values
y = dataset.iloc[ :, -1].values


print(y)