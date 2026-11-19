
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
