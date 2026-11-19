
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

for hidden_size in [(10,), (50,)]:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden_size,
            max_iter=2000,
            random_state=42
        )
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"hidden_layer_sizes = {hidden_size}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
    print()

from sklearn.ensemble import RandomForestClassifier

print("\n=== Random Forest ===")

for n in [10, 200]:
    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"n_estimators = {n}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
    print()

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

print("\n=== Model Evaluation ===")

# 使用 kNN (k=3) 進行評估
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# F1-score
print("\nMacro-F1 Score:")
print(f1_score(y_test, y_pred, average="macro"))

# 5-fold Stratified Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=3),
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro"
)

print("\n5-fold Cross-validation:")
print("Fold scores:", scores)
print("Mean Macro-F1:", scores.mean())
print("Standard Deviation:", scores.std())

