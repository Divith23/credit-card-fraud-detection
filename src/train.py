import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


#Creates the base path
ROOT = Path(__file__).resolve().parents[1]

#Distinguished paths are created
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

#Creates the directories if they do not exist
MODEL_DIR.mkdir(parents= True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents= True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

if "Class" not in df.columns:
    raise ValueError("The dataset must contain the 'Class' column.")

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Exact duplicate rows: {df.duplicated().sum()}")

df = df.drop_duplicates().reset_index(drop=True)

#Assigning features and target variable
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

#checking if the target variable contains both classes (0 and 1)
if set(y.unique()) != {0,1}:
    raise ValueError("The target must contain both classes: 0 and 1.")

#Checking if all features are numeric
if len(X.select_dtypes(exclude="number").columns) > 0:
    raise ValueError("All model input features must be numeric.")


print("Dataset after duplicate handling:", df.shape)
print(f"Feature count: {X.shape[1]}")
print("Target distribution:")
print(y.value_counts())


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


#Create a baseline model
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train,y_train)

baseline_scores = baseline.predict_proba(X_val)[:,1]

baseline_ap = average_precision_score(
    y_val,
    baseline_scores,
)


print("Baseline average precision:", baseline_ap)



logistic_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=2000,random_state=42,),),
])


random_forest_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "model",
        RandomForestClassifier(
            n_estimators=50,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])



models = {
    "logistic_regression": logistic_pipeline,
    "random_forest": random_forest_pipeline,
}

comparison_rows = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    validation_scores = model.predict_proba(X_val)[:, 1]

    ap = average_precision_score(
        y_val,
        validation_scores,
    )

    comparison_rows.append({
        "model": name,
        "validation_average_precision": ap,
    })

    print(f"{name} average precision: {ap:.6f}")

comparison_df = pd.DataFrame(comparison_rows)

print("\nModel comparison:")
print(comparison_df)

comparison_df.to_csv(
    ARTIFACTS_DIR / "model_comparison.csv",
    index=False,
)

best_name = comparison_df.loc[
    comparison_df["validation_average_precision"].idxmax(),
    "model",
]

best_model = models[best_name]

print("\nSelected model:", best_name)

print("\nGenerating validation predictions...")
best_validation_scores = best_model.predict_proba(X_val)[:, 1]
print("Validation predictions completed.")


print("\nCalculating precision-recall curve...")
precision_values, recall_values, thresholds = precision_recall_curve(
    y_val,
    best_validation_scores,
)
print("\nPrecision-recall curve calculated.")

plt.figure(figsize=(8, 5))

plt.plot(recall_values, precision_values)

plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)

plt.savefig(
    ARTIFACTS_DIR / "precision_recall_curve.png",
    bbox_inches="tight",
)


plt.close()



thresholds = np.linspace(0.01, 0.99, 99)

threshold_results = []

for threshold in thresholds:
    predictions = (best_validation_scores >= threshold).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nThreshold analysis:")
print(
    threshold_df[
        threshold_df["precision"] >= 0.90
    ].sort_values(
        "recall",
        ascending=False,
    ).head(10)
)

eligible = threshold_df[
    threshold_df["precision"] >= 0.90
]

if eligible.empty:
    print(
        "No threshold achieved 90% precision. "
        "Falling back to maximum F1."
    )

    best_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

else:
    best_row = eligible.loc[
        eligible["recall"].idxmax()
    ]

best_threshold = float(best_row["threshold"])

print("\nSelected threshold:")
print(f"Threshold : {best_threshold:.2f}")
print(f"Precision : {best_row['precision']:.4f}")
print(f"Recall    : {best_row['recall']:.4f}")
print(f"F1 Score  : {best_row['f1']:.4f}")

validation_predictions = (
    best_validation_scores >= best_threshold
).astype(int)

print(
    "Validation precision:",
    precision_score(
        y_val,
        validation_predictions,
        zero_division=0,
    ),
)

print(
    "Validation recall:",
    recall_score(
        y_val,
        validation_predictions,
        zero_division=0,
    ),
)


print("\nGenerating test predictions...")
test_scores = best_model.predict_proba(X_test)[:, 1]
print("Test predictions completed.")

test_predictions = (
    test_scores >= best_threshold
).astype(int)

test_ap = average_precision_score(
    y_test,
    test_scores,
)

print("\nFINAL TEST RESULTS")
print("Selected model:", best_name)
print("Threshold:", best_threshold)
print("Test average precision:", test_ap)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
))


model_path = MODEL_DIR / "fraud_pipeline.joblib"
metadata_path = MODEL_DIR / "model_metadata.json"

joblib.dump(best_model, model_path)

metadata = {
    "model_name": best_name,
    "threshold": float(best_threshold),
    "expected_features": list(X.columns),
    "validation_average_precision": float(
        comparison_df.loc[
            comparison_df["model"] == best_name,
            "validation_average_precision",
        ].iloc[0]
    ),
    "test_average_precision": float(test_ap),
    "test_precision": float(
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "test_recall": float(
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "test_f1": float(
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nModel saved to:", model_path)
print("Metadata saved to:", metadata_path)