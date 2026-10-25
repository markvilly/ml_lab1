# ml_lab1

Practice labs for machine learning in Python: linear regression and gradient descent, using the DeepLearning.AI ML Specialization helper files, plus data preprocessing with pandas and scikit-learn.

This guide shows how to set up a new project like this one from scratch on macOS.

## What you need

- Python 3 (for example `brew install python`)
- VS Code with the **Python** extension

## 1. Create the project folder

```bash
mkdir -p ~/Projects/my-ml-project
cd ~/Projects/my-ml-project
git init
```

## 2. Create a virtual environment

A virtual environment (`.venv`) keeps this project's libraries separate from your system Python. Homebrew's Python won't let you `pip install` into it directly anyway.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. Run `source .venv/bin/activate` again each time you open a new terminal.

## 3. Install the libraries

```bash
pip install numpy pandas matplotlib scikit-learn ipywidgets
```

| Package | Used for |
|---|---|
| `numpy` | arrays and maths |
| `pandas` | loading and slicing CSV data |
| `matplotlib` | plotting |
| `scikit-learn` | ML tools, e.g. `SimpleImputer` (you install it as `scikit-learn` but import it as `sklearn`) |
| `ipywidgets` | needed by the course's `lab_utils_uni.py` |

Record the exact versions so you can recreate the environment later:

```bash
pip freeze > requirements.txt
```

To set up on another machine or after deleting `.venv`, run `pip install -r requirements.txt`.

## 4. Point VS Code at the environment

1. Open the folder in VS Code.
2. Open the Command Palette (Cmd+Shift+P) and choose **Python: Select Interpreter**.
3. Pick the one in `.venv`.

Without this, VS Code underlines imports as missing even when they're installed.

## 5. Add the course helper files (optional)

The ML Specialization labs import `lab_utils_uni.py`, `lab_utils_common.py` and `deeplearning.mplstyle`. These aren't pip packages, so download them from the course:

1. Open a lab in the course's Jupyter environment.
2. Go to **File → Open** to see the lab folder.
3. Tick the three files and click **Download**.
4. Put them in your project folder, next to your scripts.

Then load the plot style at the top of your script:

```python
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
```

The `./` path is relative to where you run the script from, so run scripts from the project folder.

## 6. Add data

Put CSV files in the project folder and load them with pandas:

```python
import pandas as pd

dataset = pd.read_csv('pima-indians-diabetes.csv')
x = dataset.iloc[:, :-1].to_numpy()   # all rows, every column except the last (features)
y = dataset.iloc[:, -1].to_numpy()    # all rows, last column only (target)
```

## 7. Ignore files that shouldn't be committed

Create `.gitignore`:

```
.venv/
__pycache__/
.DS_Store
```

## 8. Run a script

```bash
source .venv/bin/activate
python ml_class.py
```

Or skip activating and run `.venv/bin/python ml_class.py`.

## Common problems

| Error or symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'matplotlib'` (or `sklearn`, `pandas`) | Running the system `python3` instead of `.venv` | Activate `.venv` first, or check the VS Code interpreter |
| `ModuleNotFoundError: No module named 'sklearn'` inside `.venv` | Not installed | `pip install scikit-learn` (not `pip install sklearn`) |
| `No module named 'ipywidgets'` when importing `lab_utils_uni` | The course helper needs it | `pip install ipywidgets` |
| `No module named 'lab_utils_uni'` | Helper file isn't in the project folder | Download it (step 5) |
| `TypeError: 'module' object is not callable` on `plt.style(...)` | Missing `.use` | `plt.style.use('./deeplearning.mplstyle')` |
| `InvalidIndexError` on `dataset[:, :-1]` | Slicing a pandas DataFrame like a NumPy array | Use `dataset.iloc[:, :-1]` |
| Numbers print like `5.43e+02` | NumPy's scientific notation (5.43 × 10² = 543) | `np.set_printoptions(suppress=True)` |
| A script runs old code, or a file looks different from disk | Unsaved changes in the editor | Save with Cmd+S before running |
