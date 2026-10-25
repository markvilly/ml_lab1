import numpy as np
import matplotlib.pyplot as plt

age = np.array([25, 32, 41, 47, 53, 58, 64, 70], dtype=float) # x
bp = np.array([118, 121, 127, 130, 134, 138, 142, 147], dtype=float) # y mmHg


#cost function 
def compute_cost(x,y,w,b):
    m = x.shape[0]
    total = 0

    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        total = total + error ** 2

    return total/(2*m)

# 

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        error = (w * x[i] + b) - y[i]
        dj_dw = dj_dw + error * x[i]
        dj_db = dj_db + error


    return dj_dw/m, dj_db/m

# Gradient Descent 

def gradient_descent(x, y, w_init, b_init, alpha, num_iters):

    w = w_init
    b = b_init
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost  = compute_cost (x, y, w, b)
        J_history.append(cost)

        if cost > 1e10:
            print(f"Divergent at iteration {i}")
            break

    return w, b, J_history


# Test three Learning rates

for alpha in [0.001, 0.0005, 0.0001]:
    w, b, J_hist = gradient_descent(age, bp, 0, 0, alpha, 1000 )
    print( f"alpha  = {alpha}: w = {w:.4f}, b = {b:.4f}, final J = {J_hist[-1]:.2f}")
    plt.plot(J_hist,label=f"alpha = {alpha}")


plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("J cost")
plt.title("Cost per iteration")
plt.legend()
plt.show()




# plt.scatter(age, bp, marker="x", c="r")
# plt.title("Age against Systolic blood pressure")
# plt.xlabel("Age (years)")
# plt.ylabel("BP (mmHg)")
# plt.show()

# Test


x_test = np.array([1.0, 2.0, 3.0])
y_test = np.array([2.0, 4.0, 6.0])

dw, db = compute_gradient(x_test, y_test, 1, 0)


print("Cost test: ", compute_cost(x_test, y_test,1, 0))
print("Gradient test: ", dw, ", ", db)
