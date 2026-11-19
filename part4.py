
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
