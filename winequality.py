import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# dataset = pd.read_csv('winequality-red.csv', sep=';')
# feature = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

# X = dataset[feature]
# y = dataset['quality']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# print(X_train)
# print(X_test)



import pandas as pd

dataset = pd.DataFrame({
    'fixed acidity':    [7.4, 7.8, 7.8, 6.7, 5.6, 8.9, 7.5, 7.3, 7.8, 8.5],
    'volatile acidity': [0.70, 0.88, 0.76, 0.58, 0.615, 0.62, 0.50, 0.65, 0.58, 0.28],
    'alcohol':          [9.4, 9.8, 9.8, 9.2, 9.9, 9.2, 10.5, 10.0, 9.5, 10.5],
    'quality':          [5, 5, 5, 5, 5, 5, 5, 7, 7, 7]
})

features = ['fixed acidity', 'volatile acidity', 'alcohol']

X = dataset[features]
y = dataset['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1, stratify=y)

print(y_test)