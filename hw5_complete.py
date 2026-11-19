"""Data Mining HW5 - complete reproducible experiments (Parts 1-5).

Run: python hw5_complete.py
Dependencies: numpy, pandas, matplotlib, scikit-learn
Run from the directory where you want PNG charts saved.
"""
import matplotlib
matplotlib.use("Agg")  # Save figures without blocking on GUI windows.

# ========================================================================
# PART 1
# ========================================================================


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# 載入資料集
iris = load_iris()
X, y = iris.data, iris.target

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 測試兩組不同的 K 值
for k in [3, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"K = {k}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
    print()

from sklearn.naive_bayes import GaussianNB

print("\n=== Gaussian Naive Bayes ===")

for smoothing in [1e-9, 1e-1]:
    model = GaussianNB(var_smoothing=smoothing)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"var_smoothing = {smoothing}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
    print()

from sklearn.tree import DecisionTreeClassifier

print("\n=== Decision Tree ===")

for depth in [2, 5]:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"max_depth = {depth}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
    print()

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("\n=== SVM ===")

for c in [0.1, 10]:
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=c)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"C = {c}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
    print()

from sklearn.neural_network import MLPClassifier

print("\n=== ANN (MLPClassifier) ===")