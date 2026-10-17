"""
Minimal re-implementation of the plotting helpers from the ML Specialization
lab file `lab_utils_uni.py` (univariate linear regression: f(x) = w*x + b).
"""
import numpy as np
import matplotlib.pyplot as plt


def plt_house_x(X, y, f_wb=None, ax=None):
    """Scatter the training data (size vs price) and optionally the model's prediction line."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.scatter(X, y, marker="x", c="r", label="Actual Value")
    ax.set_title("Housing Prices")
    ax.set_ylabel("Price (in 1000s of dollars)")
    ax.set_xlabel("Size (1000 sqft)")
    if f_wb is not None:
        ax.plot(X, f_wb, c="b", label="Our Prediction")
    ax.legend()


def add_line(dj_dx, x1, y1, d, ax):
    """Draw the tangent line with slope dj_dx through (x1, y1), spanning x1 +/- d."""
    x = np.linspace(x1 - d, x1 + d, 50)
    y = dj_dx * (x - x1) + y1
    ax.scatter(x1, y1, color="b", s=50)
    ax.plot(x, y, "--", c="r", zorder=10, linewidth=1)
    ax.annotate(f"dJ/dw = {dj_dx:0.0f}", xy=(x1, y1), xytext=(x1 + 20, y1),
                fontsize=10, ha="left", va="center")


def plt_gradients(x_train, y_train, f_compute_cost, f_compute_gradient, fix_b=100):
    """
    Visualise the gradient of the cost function.

    f_compute_cost(x, y, w, b)     -> scalar cost J(w,b)
    f_compute_gradient(x, y, w, b) -> (dj_dw, dj_db)

    Left:  J(w) with b held fixed, plus tangent lines whose slope is dJ/dw at a few w values.
    Right: quiver plot of the gradient vector (dJ/dw, dJ/db) over a grid of (w, b).
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left: cost curve along w with gradient tangents ---
    w_array = np.linspace(-100, 500, 50)
    cost = np.array([f_compute_cost(x_train, y_train, w, fix_b) for w in w_array])
    ax[0].plot(w_array, cost, linewidth=1)
    ax[0].set_title(f"Cost vs w, with gradient; b set to {fix_b}")
    ax[0].set_ylabel("Cost")
    ax[0].set_xlabel("w")

    for w in [100, 200, 300]:
        dj_dw, _ = f_compute_gradient(x_train, y_train, w, fix_b)
        j = f_compute_cost(x_train, y_train, w, fix_b)
        add_line(dj_dw, w, j, 30, ax[0])

    # --- Right: gradient field over (w, b) ---
    tmp_b, tmp_w = np.meshgrid(np.linspace(-200, 200, 10), np.linspace(-100, 600, 10))
    U = np.zeros_like(tmp_w)  # dJ/dw
    V = np.zeros_like(tmp_b)  # dJ/db
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            U[i, j], V[i, j] = f_compute_gradient(x_train, y_train, tmp_w[i, j], tmp_b[i, j])

    magnitude = np.sqrt(U ** 2 + V ** 2)
    ax[1].quiver(tmp_w, tmp_b, U, V, magnitude, units="width")
    ax[1].set_title("Gradient shown in quiver plot")
    ax[1].set_xlabel("w")
    ax[1].set_ylabel("b")

    plt.tight_layout()
    plt.show()
