from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

ML_DATA = BASE_DIR / "data" / "ml_ready" / "phase1_event_level_ml_dataset.csv"

SOURCE_CANDIDATES = (
    BASE_DIR
    / "data"
    / "ml_ready"
    / "supervised_source_candidates.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "lightgbm_source_classifier.txt"
FEATURE_FILE = MODEL_DIR / "prototype_features.json"
LABEL_FILE = MODEL_DIR / "label_mapping.json"
METADATA_FILE = MODEL_DIR / "source_model_metadata.json"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "observation_count",
    "mean_frp",
    "peak_frp",
    "total_frp",
    "mean_brightness",
    "persistent",
    "historical_detection_count",
    "historical_mean_frp",
    "local_frp_deviation",
    "historical_daily_activity",
    "previously_detected",
    "anomaly_score",
    "is_anomaly",
]


# ============================================================
# LABEL MAPPING
# ============================================================

LABELS = [
    "Agriculture_Biomass",
    "Forest_Natural",
    "Industrial",
    "Waste_Other",
]

LABEL_TO_ID = {
    label: i
    for i, label in enumerate(LABELS)
}

ID_TO_LABEL = {
    str(i): label
    for label, i in LABEL_TO_ID.items()
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FIRE-SIGHT AI — SOURCE CLASSIFIER TRAINING")
print("=" * 70)

print("\n[1/8] Loading ML dataset...")

df = pd.read_csv(ML_DATA)

print("ML dataset:", df.shape)

if "event_id" not in df.columns:
    raise ValueError(
        "event_id column is missing from phase1_event_level_ml_dataset.csv"
    )


print("\n[2/8] Loading source candidate labels...")

labels = pd.read_csv(SOURCE_CANDIDATES)

print("Source candidate dataset:", labels.shape)

print("Candidate columns:")
print(labels.columns.tolist())


# ============================================================
# FIND LABEL COLUMN
# ============================================================

possible_label_columns = [
    "source_candidate",
    "verified_label",
    "label",
    "source",
]

label_column = None

for column in possible_label_columns:
    if column in labels.columns:
        label_column = column
        break

if label_column is None:
    raise ValueError(
        "Could not find source label column. "
        "Expected one of: "
        + ", ".join(possible_label_columns)
    )

print("Using label column:", label_column)


# ============================================================
# MERGE
# ============================================================

print("\n[3/8] Merging event features with labels...")

labels = labels[["event_id", label_column]].copy()

labels = labels.drop_duplicates(
    subset=["event_id"]
)

train_df = df.merge(
    labels,
    on="event_id",
    how="inner",
)

print("Merged dataset:", train_df.shape)


# ============================================================
# CLEAN LABELS
# ============================================================

train_df[label_column] = (
    train_df[label_column]
    .astype(str)
    .str.strip()
)

train_df = train_df[
    train_df[label_column].isin(LABELS)
].copy()

print("\nLabel distribution:")
print(train_df[label_column].value_counts())


# ============================================================
# CHECK FEATURES
# ============================================================

print("\n[4/8] Checking features...")

missing_features = [
    feature
    for feature in FEATURES
    if feature not in train_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )

X = train_df[FEATURES].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

y_text = train_df[label_column]

y = y_text.map(LABEL_TO_ID).astype(int)


# ============================================================
# SPATIAL GROUP
# ============================================================

print("\n[5/8] Creating spatial groups...")

GRID_SIZE = 0.1

train_df["spatial_group"] = (
    (
        train_df["latitude"] / GRID_SIZE
    )
    .apply(np.floor)
    .astype(int)
    .astype(str)
    + "_"
    +
    (
        train_df["longitude"] / GRID_SIZE
    )
    .apply(np.floor)
    .astype(int)
    .astype(str)
)

groups = train_df["spatial_group"]


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

print("\n[6/8] Splitting spatially...")

gss1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, temp_idx = next(
    gss1.split(
        X,
        y,
        groups=groups,
    )
)

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_temp = X.iloc[temp_idx]
y_temp = y.iloc[temp_idx]

groups_temp = groups.iloc[temp_idx]


gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42,
)

val_idx, test_idx = next(
    gss2.split(
        X_temp,
        y_temp,
        groups=groups_temp,
    )
)

X_val = X_temp.iloc[val_idx]
y_val = y_temp.iloc[val_idx]

X_test = X_temp.iloc[test_idx]
y_test = y_temp.iloc[test_idx]


print("\nSplit sizes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_counts = y_train.value_counts().sort_index()

total = len(y_train)
n_classes = len(LABELS)

class_weights = {}

for class_id in range(n_classes):

    count = class_counts.get(
        class_id,
        1
    )

    class_weights[class_id] = (
        total / (n_classes * count)
    )

print("\nClass weights:")

for class_id, weight in class_weights.items():
    print(
        f"{ID_TO_LABEL[str(class_id)]}: "
        f"{weight:.4f}"
    )


sample_weights = y_train.map(
    class_weights
).values


# ============================================================
# LIGHTGBM
# ============================================================

print("\nTraining LightGBM...")

model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=n_classes,

    n_estimators=300,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1,

    verbosity=-1,
)


model.fit(
    X_train,
    y_train,

    sample_weight=sample_weights,

    eval_set=[
        (X_val, y_val)
    ],

    eval_metric="multi_logloss",

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=30,
            verbose=True
        )
    ],
)


# ============================================================
# EVALUATION
# ============================================================

print("\n[7/8] Evaluating model...")

pred = model.predict(X_test)

pred = np.asarray(pred).astype(int).ravel()

accuracy = accuracy_score(
    y_test,
    pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    pred
)

print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(
    f"Accuracy:           {accuracy:.4f}"
)

print(
    f"Balanced Accuracy:  {balanced_accuracy:.4f}"
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        pred,
        labels=list(range(n_classes)),
        target_names=LABELS,
        zero_division=0,
    )
)

print("\nConfusion matrix:")

print(
    confusion_matrix(
        y_test,
        pred,
        labels=list(range(n_classes)),
    )
)


# ============================================================
# SAVE NATIVE LIGHTGBM MODEL
# ============================================================

print("\n[8/8] Saving model...")

booster = model.booster_

booster.save_model(
    str(MODEL_FILE)
)

print(
    "Saved native LightGBM model:"
)

print(
    MODEL_FILE
)


# ============================================================
# SAVE FEATURE CONFIG
# ============================================================

with open(
    FEATURE_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        FEATURES,
        f,
        indent=4,
    )


# ============================================================
# SAVE LABEL CONFIG
# ============================================================

with open(
    LABEL_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "classes": LABELS,
        },
        f,
        indent=4,
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model_type": "LightGBM",
    "model_format": "native_lightgbm_txt",
    "lightgbm_version": lgb.__version__,
    "features": FEATURES,
    "classes": LABELS,
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(
        balanced_accuracy
    ),
    "random_state": 42,
    "label_source": str(SOURCE_CANDIDATES),
    "label_column": label_column,
}

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(
    f"1. {MODEL_FILE}"
)

print(
    f"2. {FEATURE_FILE}"
)

print(
    f"3. {LABEL_FILE}"
)

print(
    f"4. {METADATA_FILE}"
)

print("\nThe old corrupted .pkl is no longer required.")