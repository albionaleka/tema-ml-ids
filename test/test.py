from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# 1. PATHS
# ============================================================

MODEL_DIR = Path(
    "/home/user/Documents/project/models"
)

ARTIFACT_DIR = Path(
    "/home/user/Documents/project/artifacts/random_forest"
)


# ============================================================
# 2. FILE PATHS
# ============================================================

MODEL_PATH = (
    MODEL_DIR /
    "random_forest.joblib"
)

FEATURES_PATH = (
    ARTIFACT_DIR /
    "selected_features.pkl"
)

IMPUTER_PATH = (
    ARTIFACT_DIR /
    "imputer.pkl"
)

ENCODER_PATH = (
    ARTIFACT_DIR /
    "label_encoder.pkl"
)


# ============================================================
# 3. LOAD MODEL AND ARTIFACTS
# ============================================================

print("=" * 70)
print("LOADING RANDOM FOREST MODEL")
print("=" * 70)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Feature list not found:\n{FEATURES_PATH}"
    )


if not IMPUTER_PATH.exists():
    raise FileNotFoundError(
        f"Imputer not found:\n{IMPUTER_PATH}"
    )


if not ENCODER_PATH.exists():
    raise FileNotFoundError(
        f"Label encoder not found:\n{ENCODER_PATH}"
    )


model = joblib.load(
    MODEL_PATH
)

feature_names = joblib.load(
    FEATURES_PATH
)

imputer = joblib.load(
    IMPUTER_PATH
)

encoder = joblib.load(
    ENCODER_PATH
)


print("Model loaded successfully.")

print(
    f"Number of expected features: "
    f"{len(feature_names)}"
)


# ============================================================
# 4. DISPLAY CLASSES
# ============================================================

print("\nClasses:")

for index, label in enumerate(
    encoder.classes_
):
    print(
        f"{index}: {label}"
    )


# ============================================================
# 5. DISPLAY FEATURES
# ============================================================

print("\n")
print("=" * 70)
print("MODEL FEATURES")
print("=" * 70)

for index, feature in enumerate(
    feature_names
):
    print(
        f"{index:02d}: {feature}"
    )


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict_network_traffic(**features):

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in features
    ]

    if missing_features:

        raise ValueError(
            "Missing required features:\n"
            + "\n".join(
                f" - {feature}"
                for feature in missing_features
            )
        )


    # --------------------------------------------------------
    # Check extra features
    # --------------------------------------------------------

    extra_features = [
        feature
        for feature in features
        if feature not in feature_names
    ]

    if extra_features:

        print("\nWARNING:")
        print(
            "The following supplied features "
            "are not used by the model:"
        )

        for feature in extra_features:
            print(
                f" - {feature}"
            )


    # --------------------------------------------------------
    # Create DataFrame
    #
    # IMPORTANT:
    # Use EXACT same feature order as training.
    # --------------------------------------------------------

    data = pd.DataFrame(
        [
            [
                features[feature]
                for feature in feature_names
            ]
        ],
        columns=feature_names
    )


    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # --------------------------------------------------------
    # Replace infinity values
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # Apply saved median imputer
    # --------------------------------------------------------

    data_imputed = imputer.transform(
        data
    )


    # --------------------------------------------------------
    # Predict encoded class
    # --------------------------------------------------------

    prediction_encoded = model.predict(
        data_imputed
    )[0]


    # --------------------------------------------------------
    # Decode class
    # --------------------------------------------------------

    prediction = encoder.inverse_transform(
        [prediction_encoded]
    )[0]


    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        data_imputed
    )[0]


    probability_dict = {
        label: float(probability)
        for label, probability in zip(
            encoder.classes_,
            probabilities
        )
    }


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        np.max(probabilities)
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_dict
    }


# ============================================================
# 7. CIC-IDS2017 SAMPLE
# ============================================================

sample = {

    "Destination Port": 2170,

    "Flow Duration": 77,

    "Total Fwd Packets": 1,

    "Total Backward Packets": 1,

    "Total Length of Fwd Packets": 0,

    "Total Length of Bwd Packets": 6,

    "Fwd Packet Length Max": 0,

    "Fwd Packet Length Min": 0,

    "Fwd Packet Length Mean": 0,

    "Fwd Packet Length Std": 0,

    "Bwd Packet Length Max": 6,

    "Bwd Packet Length Min": 6,

    "Bwd Packet Length Mean": 6,

    "Bwd Packet Length Std": 0,

    "Flow Bytes/s": 77922.07792,

    "Flow Packets/s": 25974.02597,

    "Flow IAT Mean": 77,

    "Flow IAT Std": 0,

    "Flow IAT Max": 77,

    "Flow IAT Min": 77,

    "Fwd IAT Total": 0,

    "Fwd IAT Mean": 0,

    "Fwd IAT Std": 0,

    "Fwd IAT Max": 0,

    "Fwd IAT Min": 0,

    "Bwd IAT Total": 0,

    "Bwd IAT Mean": 0,

    "Bwd IAT Std": 0,

    "Bwd IAT Max": 0,

    "Bwd IAT Min": 0,

    "Fwd PSH Flags": 0,

    "Bwd PSH Flags": 0,

    "Fwd URG Flags": 0,

    "Bwd URG Flags": 0,

    "Fwd Header Length": 40,

    "Bwd Header Length": 20,

    "Fwd Packets/s": 12987.01299,

    "Bwd Packets/s": 12987.01299,

    "Min Packet Length": 0,

    "Max Packet Length": 6,

    "Packet Length Mean": 2,

    "Packet Length Std": 3.464101615,

    "Packet Length Variance": 12,

    "FIN Flag Count": 0,

    "SYN Flag Count": 0,

    "RST Flag Count": 0,

    "PSH Flag Count": 1,

    "ACK Flag Count": 0,

    "URG Flag Count": 0,

    "CWE Flag Count": 0,

    "ECE Flag Count": 0,

    "Down/Up Ratio": 1,

    "Average Packet Size": 3,

    "Avg Fwd Segment Size": 0,

    "Avg Bwd Segment Size": 6,

    # IMPORTANT:
    # Your training dataset had duplicate
    # "Fwd Header Length".
    #
    # Pandas renamed the second one:
    # "Fwd Header Length.1"

    "Fwd Header Length.1": 40,

    "Fwd Avg Bytes/Bulk": 0,

    "Fwd Avg Packets/Bulk": 0,

    "Fwd Avg Bulk Rate": 0,

    "Bwd Avg Bytes/Bulk": 0,

    "Bwd Avg Packets/Bulk": 0,

    "Bwd Avg Bulk Rate": 0,

    "Subflow Fwd Packets": 1,

    "Subflow Fwd Bytes": 0,

    "Subflow Bwd Packets": 1,

    "Subflow Bwd Bytes": 6,

    "Init_Win_bytes_forward": 29200,

    "Init_Win_bytes_backward": 0,

    "act_data_pkt_fwd": 0,

    "min_seg_size_forward": 40,

    "Active Mean": 0,

    "Active Std": 0,

    "Active Max": 0,

    "Active Min": 0,

    "Idle Mean": 0,

    "Idle Std": 0,

    "Idle Max": 0,

    "Idle Min": 0
}


# ============================================================
# 8. RUN PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("NETWORK TRAFFIC PREDICTION")
print("=" * 70)


result = predict_network_traffic(
    **sample
)


# ============================================================
# 9. EXTRACT RESULT
# ============================================================

prediction = result[
    "prediction"
]

confidence = result[
    "confidence"
]

probabilities = result[
    "probabilities"
]


# ============================================================
# 10. DISPLAY PREDICTION
# ============================================================

print("\nPrediction:")
print(
    f"  {prediction}"
)


print("\nConfidence:")
print(
    f"  {confidence:.2%}"
)


# ============================================================
# 11. DISPLAY PROBABILITIES
# ============================================================

print("\nClass probabilities:")


sorted_probabilities = sorted(
    probabilities.items(),
    key=lambda item: item[1],
    reverse=True
)


for label, probability in (
    sorted_probabilities
):

    print(
        f"  {label:<20}"
        f"{probability:.2%}"
    )


# ============================================================
# 12. FINAL DECISION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DECISION")
print("=" * 70)


if prediction == "BENIGN":

    print(
        "Traffic classification: BENIGN"
    )

else:

    print(
        "Traffic classification: ATTACK"
    )

    print(
        f"Attack category: {prediction}"
    )


# ============================================================
# 13. RESULT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("RESULT SUMMARY")
print("=" * 70)

print(
    f"Prediction : {prediction}"
)

print(
    f"Confidence : {confidence:.2%}"
)

print(
    f"Features   : {len(feature_names)}"
)

print(
    f"Model      : Random Forest"
)

print(
    f"Model path : {MODEL_PATH}"
)