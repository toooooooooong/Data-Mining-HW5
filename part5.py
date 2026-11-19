
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

best_name = results_df.iloc[0]["Classifier"]
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
