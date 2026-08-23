from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

from pathlib import Path

DATA_DIR = Path("/home/user/Documents/project/data/cicids2017")

MODEL_DIR = Path("/home/user/Documents/project/models/extra_trees")
ARTIFACT_DIR = Path("/home/user/Documents/project/artifacts/extra_trees")
PLOT_DIR = Path("/home/user/Documents/project/plots/extra_trees")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# 2. CIC-IDS2017 LABEL MAPPING

LABEL_MAPPING = {

    "BENIGN": "BENIGN",

    "DoS Hulk": "DOS",
    "DoS GoldenEye": "DOS",
    "DoS slowloris": "DOS",
    "DoS Slowhttptest": "DOS",

    "DDoS": "DDOS",

    "PortScan": "PORTSCAN",

    "FTP-Patator": "BRUTE_FORCE",
    "SSH-Patator": "BRUTE_FORCE",

    "Web Attack - Brute Force": "WEB_ATTACK",
    "Web Attack - XSS": "WEB_ATTACK",
    "Web Attack - Sql Injection": "WEB_ATTACK",

    "Bot": "BOTNET",

    "Infiltration": "INFILTRATION",

    "Heartbleed": "HEARTBLEED",
}

# 3. CLEAN COLUMN NAMES

def clean_column_names(df):

    df = df.copy()

    df.columns = [
        " ".join(
            str(c)
            .replace("\ufeff", "")
            .strip()
            .split()
        )
        for c in df.columns
    ]

    return df

# 4. LOAD DATA

def load_dataset(data_dir):

    files = sorted(
        Path(data_dir).rglob("*.csv")
    )

    if not files:

        raise FileNotFoundError(
            f"No CSV files found in "
            f"{Path(data_dir).resolve()}"
        )

    frames = []

    print("=" * 70)
    print("LOADING CIC-IDS2017")
    print("=" * 70)

    for file in files:

        print(
            f"Loading: {file.name}"
        )

        df = pd.read_csv(
            file,
            low_memory=False
        )

        df = clean_column_names(
            df
        )

        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    data = clean_column_names(
        data
    )

    print(
        f"\nDataset shape: {data.shape}"
    )

    return data


# 5. FIND LABEL

def find_label_column(df):

    for name in [
        "Label",
        "label",
        "LABEL"
    ]:

        if name in df.columns:
            return name

    raise ValueError(
        "Label column not found."
    )


# 6. NORMALIZE LABELS

def normalize_labels(
    df,
    label_column
):

    df = df.copy()

    raw = (
        df[label_column]
        .astype(str)
        .str.strip()
    )

    def normalize(label):

        if label.upper() == "BENIGN":
            return "BENIGN"

        return LABEL_MAPPING.get(
            label,
            "OTHER_ATTACK"
        )

    df["Attack_Category"] = (
        raw.map(normalize)
    )

    print("\nNormalized classes:")
    print(
        df["Attack_Category"]
        .value_counts()
    )

    return df

# 7. FEATURE SELECTION

def select_features(
    df,
    label_column
):

    excluded = {

        "Flow ID",
        "FlowID",

        "Source IP",
        "Destination IP",

        "Timestamp",

        label_column,

        "Attack_Category"

    }

    candidates = [
        c
        for c in df.columns
        if c not in excluded
    ]

    numeric = df[
        candidates
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    usable = [
        c
        for c in numeric.columns
        if numeric[c].notna().any()
    ]

    X = numeric[
        usable
    ].copy()

    feature_names = list(
        X.columns
    )

    print(
        f"\nSelected {len(feature_names)} features."
    )

    return X, feature_names

# 8. PREPARE DATA

def prepare_data():

    df = load_dataset(
        DATA_DIR
    )

    label_column = find_label_column(
        df
    )

    df = normalize_labels(
        df,
        label_column
    )

    X, feature_names = select_features(
        df,
        label_column
    )

    y = df[
        "Attack_Category"
    ]

    valid_rows = ~X.isna().all(
        axis=1
    )

    X = X.loc[
        valid_rows
    ]

    y = y.loc[
        valid_rows
    ]

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        y
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,
        y_encoded,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y_encoded
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        imputer,
        encoder
    )

    
# 9. PREPARE

(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    imputer,
    encoder
) = prepare_data()

# 10. CREATE EXTRA TREES

print("=" * 70)
print("TRAINING EXTRA TREES")
print("=" * 70)

model = ExtraTreesClassifier(

    n_estimators=300,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    max_features="sqrt"
)

# 11. TRAIN

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

print("Training completed.")

# 12. PREDICT

y_pred = model.predict(
    X_test
)

# 13. METRICS

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

try:

    probabilities = model.predict_proba(
        X_test
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="weighted"
    )

except Exception:

    roc_auc = np.nan

# 14. RESULTS

print("\n")
print("=" * 70)
print("EXTRA TREES RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

# 15. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(11, 9)
)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
).plot(
    ax=ax,
    xticks_rotation=45,
    colorbar=False
)

ax.set_title(
    "Extra Trees - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR /
    "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

plt.close()

# 16. METRICS PLOT

metrics = pd.Series({

    "Accuracy": accuracy,

    "Precision": precision,

    "Recall": recall,

    "F1-Score": f1,

    "ROC-AUC": roc_auc

})

fig, ax = plt.subplots(
    figsize=(9, 5)
)

metrics.plot(
    kind="bar",
    ax=ax
)

ax.set_ylim(
    0,
    1
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Extra Trees - Evaluation Metrics"
)

ax.tick_params(
    axis="x",
    rotation=30
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR /
    "evaluation_metrics.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

plt.close()

# 17. FEATURE IMPORTANCE

importance = pd.Series(

    model.feature_importances_,

    index=feature_names

).sort_values(
    ascending=False
)

top20 = importance.head(20)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

top20.sort_values().plot(
    kind="barh",
    ax=ax
)

ax.set_title(
    "Extra Trees - Top 20 Feature Importance"
)

ax.set_xlabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR /
    "feature_importance_top20.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

plt.close()

# 18. SAVE MODEL

model_path = (
    MODEL_DIR /
    "extra_trees.joblib"
)

joblib.dump(
    model,
    model_path
)

# 19. SAVE ARTIFACTS

joblib.dump(
    feature_names,
    ARTIFACT_DIR /
    "selected_features.pkl"
)

joblib.dump(
    imputer,
    ARTIFACT_DIR /
    "imputer.pkl"
)

joblib.dump(
    encoder,
    ARTIFACT_DIR /
    "label_encoder.pkl"
)

# 20. PREPROCESSING BUNDLE

bundle = {

    "feature_names":
        feature_names,

    "imputer":
        imputer,

    "label_encoder":
        encoder,

    "label_mapping":
        LABEL_MAPPING

}

joblib.dump(
    bundle,
    ARTIFACT_DIR /
    "preprocessing.pkl"
)

# 21. SAVE METRICS

results = pd.DataFrame([{

    "Model":
        "Extra Trees",

    "Accuracy":
        accuracy,

    "Precision":
        precision,

    "Recall":
        recall,

    "F1-Score":
        f1,

    "ROC-AUC":
        roc_auc

}])

results.to_csv(
    ARTIFACT_DIR /
    "metrics.csv",
    index=False
)

# 22. SAVE FEATURE LIST

pd.DataFrame({

    "feature_index":
        range(len(feature_names)),

    "feature_name":
        feature_names

}).to_csv(
    ARTIFACT_DIR /
    "selected_features.csv",
    index=False
)


print("\n")
print("=" * 70)
print("EXTRA TREES COMPLETE")
print("=" * 70)

print(
    f"Model: {model_path}"
)

print(
    f"Plots: {PLOT_DIR}"
)