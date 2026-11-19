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



# ========================================================================
# PART 2
# ========================================================================


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd

# Q4: Dataset information
data = load_breast_cancer()
X, y = data.data, data.target

print("=== Q4: Dataset Information ===")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Class names:", data.target_names)

for i, name in enumerate(data.target_names):
    print(f"{name}: {(y == i).sum()} samples")

feature_ranges = pd.DataFrame({
    "Feature": data.feature_names,
    "Min": X.min(axis=0),
    "Max": X.max(axis=0)
})
print("\nFeature ranges:")
print(feature_ranges.to_string(index=False))

# Shared 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Q6: Raw baseline (NO standardization)
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "kNN": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Gaussian Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", C=1.0)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Classifier": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro-F1": f1_score(
            y_test, predictions, average="macro"
        )
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by=["Macro-F1", "Accuracy"],
    ascending=False
)

print("\n=== Q6: Raw Baseline Results ===")
print(results_df.to_string(index=False, float_format="%.4f"))


# ========================================================================
# PART 3
# ========================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Q9: Load dataset and split
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Q10: Compare raw and scaled features
def create_models():
    return {
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=1.0),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

results = []

for name in create_models():
    scores = {}

    for version, train_X, test_X in [
        ("Raw", X_train, X_test),
        ("Scaled", X_train_scaled, X_test_scaled)
    ]:
        model = create_models()[name]
        model.fit(train_X, y_train)
        predictions = model.predict(test_X)

        scores[f"{version} Acc."] = accuracy_score(
            y_test, predictions
        )
        scores[f"{version} Macro-F1"] = f1_score(
            y_test, predictions, average="macro"
        )

    results.append({
        "Classifier": name,
        **scores
    })

results_df = pd.DataFrame(results)

print("=== Part 3: Standardization Comparison ===")
print(results_df.to_string(
    index=False,
    float_format="%.4f"
))

# Q11: Create comparison chart
names = results_df["Classifier"].tolist()
x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    x - width / 2,
    results_df["Raw Macro-F1"],
    width,
    label="Raw"
)
ax.bar(
    x + width / 2,
    results_df["Scaled Macro-F1"],
    width,
    label="Standardized"
)

ax.set_xlabel("Classifier")
ax.set_ylabel("Macro-F1")
ax.set_title("Effect of Standardization on Classifiers")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    "part3_standardization.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nChart saved: part3_standardization.png")


# ========================================================================
# PART 4
# ========================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. Load dataset
# ==========================================

data = load_breast_cancer()
X, y = data.data, data.target

# Same split as Part 2 and Part 3
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5-fold Stratified Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 2. Hyperparameter candidates (Q14)
# ==========================================

candidates = {
    "kNN": [1, 3, 5, 9, 15],
    "Decision Tree": [2, 3, 5, 10, None],
    "SVM": [0.1, 1, 10]
}


def build_model(name, parameter):

    if name == "kNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=int(parameter)
            )
        )

    elif name == "Decision Tree":
        depth = (
            None if parameter is None
            else int(parameter)
        )

        return DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

    elif name == "SVM":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=float(parameter)
            )
        )

    else:
        raise ValueError(f"Unknown classifier: {name}")


# ==========================================
# 3. Cross-validation experiments (Q14-Q15)
# ==========================================

results = []

for name, parameters in candidates.items():

    for parameter in parameters:

        model = build_model(name, parameter)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro"
        )

        results.append({
            "Classifier": name,
            "Parameter": str(parameter),
            "Mean Macro-F1": scores.mean(),
            "Std": scores.std()
        })


results_df = pd.DataFrame(results)

print("\n=== Q14-Q15: CV Results ===")
print(
    results_df.to_string(
        index=False,
        float_format="%.4f"
    )
)


# ==========================================
# 4. Select best parameters (Q16)
# ==========================================

best_settings = {}

for name, parameters in candidates.items():

    subset = results_df[
        results_df["Classifier"] == name
    ]

    best_row = subset.loc[
        subset["Mean Macro-F1"].idxmax()
    ]

    # Match the selected string to the original
    # candidate, preserving int / float / None.
    selected_parameter = next(
        p for p in parameters
        if str(p) == best_row["Parameter"]
    )

    best_settings[name] = selected_parameter


print("\n=== Q16: Selected Parameters ===")

for name, parameter in best_settings.items():
    print(f"{name}: {parameter}")


# ==========================================
# 5. Hyperparameter comparison charts (Q15)
# ==========================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4.5)
)

for ax, (name, parameters) in zip(
    axes, candidates.items()
):

    subset = results_df[
        results_df["Classifier"] == name
    ]

    labels = subset["Parameter"].tolist()
    means = subset["Mean Macro-F1"].to_numpy()
    stds = subset["Std"].to_numpy()

    positions = np.arange(len(labels))

    ax.errorbar(
        positions,
        means,
        yerr=stds,
        marker="o",
        capsize=4
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    ax.set_title(name)
    ax.set_xlabel("Hyperparameter")
    ax.set_ylabel("Mean CV Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)


plt.suptitle(
    "Hyperparameter vs. CV Performance"
)

plt.tight_layout()

plt.savefig(
    "part4_cv_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# Close figure without waiting for a window
plt.close()

print("\nChart saved: part4_cv_comparison.png")


# ==========================================
# 6. Final test evaluation (Q18)
# ==========================================

print("\n=== Q18: Final Test Results ===")

final_results = []

for name, parameter in best_settings.items():

    model = build_model(name, parameter)

    # Train on the complete training set
    model.fit(X_train, y_train)

    # Evaluate on test set only after selection
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test, predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    final_results.append({
        "Classifier": name,
        "Parameter": str(parameter),
        "Accuracy": accuracy,
        "Macro-F1": macro_f1
    })


final_df = pd.DataFrame(final_results)

print(
    final_df.to_string(
        index=False,
        float_format="%.4f"
    )
)

print("\nPart 4 completed successfully!")


# ========================================================================
# PART 5
# ========================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# Q19: Dataset Information
# ==========================================

data = load_digits()
X, y = data.data, data.target

print("=== Q19: Digits Dataset Information ===")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", len(np.unique(y)))
print("Class names:", data.target_names)

print("\nClass distribution:")
for digit in np.unique(y):
    count = np.sum(y == digit)
    print(f"Digit {digit}: {count} samples")

print("\nFeature range:")
print("Minimum:", X.min())
print("Maximum:", X.max())

print("\nImage shape:", data.images.shape[1:])


# ==========================================
# Q20: 80/20 Stratified Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n=== Q20: Train/Test Split ===")
print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))


# ==========================================
# Q20-Q21: Five Classifiers
# ==========================================

# Keep baseline settings for comparison.
# StandardScaler is used for kNN and SVM.
# Tree-based models use raw features.

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "kNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Gaussian Naive Bayes": GaussianNB(),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0)
    )
}

results = []
predictions_by_model = {}

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    results.append({
        "Classifier": name,
        "Accuracy": accuracy,
        "Macro-F1": macro_f1
    })

    predictions_by_model[name] = predictions


results_df = pd.DataFrame(results)

# Rank by Macro-F1, then Accuracy
results_df = results_df.sort_values(
    by=["Macro-F1", "Accuracy"],
    ascending=False
).reset_index(drop=True)

print("\n=== Q21: Digits Classification Results ===")
print(
    results_df.to_string(
        index=False,
        float_format="%.4f"
    )
)


# ==========================================
# Q21: Best Classifier Confusion Matrix
# ==========================================

# SVM is specified for the confusion-matrix illustration; do not select via test scores.
best_name = "SVM"
best_predictions = predictions_by_model[best_name]

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=np.arange(10)
)

print("\n=== Best Classifier ===")
print(best_name)

print("\n=== Confusion Matrix ===")
print(cm)

fig, ax = plt.subplots(figsize=(9, 7))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=True
)

ax.set_title(
    f"Digits Confusion Matrix - {best_name}"
)

plt.tight_layout()

plt.savefig(
    "part5_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved: part5_confusion_matrix.png")


# ==========================================
# Additional: Classifier Comparison Chart
# ==========================================

names = results_df["Classifier"].tolist()

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))

ax.bar(
    x - width / 2,
    results_df["Accuracy"],
    width,
    label="Accuracy"
)

ax.bar(
    x + width / 2,
    results_df["Macro-F1"],
    width,
    label="Macro-F1"
)

ax.set_xlabel("Classifier")
ax.set_ylabel("Score")
ax.set_title("Digits Classifier Performance")

ax.set_xticks(x)
ax.set_xticklabels(
    names,
    rotation=15,
    ha="right"
)

ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    "part5_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: part5_comparison.png")


# ==========================================
# Finish
# ==========================================

print("\nPart 5 experiments completed successfully!")
