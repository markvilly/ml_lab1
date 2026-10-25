import numpy as np
import time

#VECTOR CREATION

# a = np.zeros(4);    print(f"np.zeros(4): a = {a}, a.shape = {a.shape}, a data type = {a.dtype}")
# a = np.zeros((4,)); print(f"np.zeros((4,)): a = {a}, a.shape = {a.shape}, a data type = {a.dtype}")
# a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a.shape = {a.shape}, a data type = {a.dtype}")


# a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


#INDEXING ELEMENTS

#vector-indexing operations

a = np.arange(10)
print(a)

try:
    c = a[10]
except Exception as e:
    print("Error you will reecieve")
    print(e)



