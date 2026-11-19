
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
plt.show()

print("\nChart saved: part3_standardization.png")
