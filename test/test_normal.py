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
# 3. LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING RANDOM FOREST MODEL")
print("=" * 70)

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

print("\nClasses:")

for index, label in enumerate(
    encoder.classes_
):
    print(
        f"{index}: {label}"
    )


# ============================================================
# 4. SHOW FEATURES
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
# 5. PREDICTION FUNCTION
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
    # Create DataFrame in exact training order
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
    # Convert values to numeric
    # --------------------------------------------------------

    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # Apply saved imputer
    # --------------------------------------------------------

    data_imputed = imputer.transform(
        data
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction_encoded = model.predict(
        data_imputed
    )[0]


    # --------------------------------------------------------
    # Decode prediction
    # --------------------------------------------------------

    prediction = encoder.inverse_transform(
        [prediction_encoded]
    )[0]


    # --------------------------------------------------------
    # Probability
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


    confidence = float(
        np.max(probabilities)
    )


    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_dict
    }


# ============================================================
# 6. YOUR NEW CICIDS2017 SAMPLE
# ============================================================

sample = {

    "Destination Port": 443,

    "Flow Duration": 119717550,

    "Total Fwd Packets": 89,

    "Total Backward Packets": 113,

    "Total Length of Fwd Packets": 1398,

    "Total Length of Bwd Packets": 141366,

    "Fwd Packet Length Max": 332,

    "Fwd Packet Length Min": 0,

    "Fwd Packet Length Mean": 15.70786517,

    "Fwd Packet Length Std": 49.89038035,

    "Bwd Packet Length Max": 2836,

    "Bwd Packet Length Min": 0,

    "Bwd Packet Length Mean": 1251.026549,

    "Bwd Packet Length Std": 514.2899484,

    "Flow Bytes/s": 1192.506863,

    "Flow Packets/s": 1.687304827,

    "Flow IAT Mean": 595609.7015,

    "Flow IAT Std": 5851363.912,

    "Flow IAT Max": 59000000,

    "Flow IAT Min": 1,

    "Fwd IAT Total": 120000000,

    "Fwd IAT Mean": 1360426.705,

    "Fwd IAT Std": 8812427.19,

    "Fwd IAT Max": 59000000,

    "Fwd IAT Min": 1,

    "Bwd IAT Total": 120000000,

    "Bwd IAT Mean": 1068578.17,

    "Bwd IAT Std": 7827246.938,

    "Bwd IAT Max": 59000000,

    "Bwd IAT Min": 1,

    "Fwd PSH Flags": 0,

    "Bwd PSH Flags": 0,

    "Fwd URG Flags": 0,

    "Bwd URG Flags": 0,

    "Fwd Header Length": 2868,

    "Bwd Header Length": 3624,

    "Fwd Packets/s": 0.743416483,

    "Bwd Packets/s": 0.943888344,

    "Min Packet Length": 0,

    "Max Packet Length": 2836,

    "Packet Length Mean": 703.270936,

    "Packet Length Std": 725.4741983,

    "Packet Length Variance": 526312.8124,

    "FIN Flag Count": 0,

    "SYN Flag Count": 0,

    "RST Flag Count": 0,

    "PSH Flag Count": 1,

    "ACK Flag Count": 0,

    "URG Flag Count": 0,

    "CWE Flag Count": 0,

    "ECE Flag Count": 0,

    "Down/Up Ratio": 1,

    "Average Packet Size": 706.7524752,

    "Avg Fwd Segment Size": 15.70786517,

    "Avg Bwd Segment Size": 1251.026549,

    "Fwd Header Length.1": 2868,

    "Fwd Avg Bytes/Bulk": 0,

    "Fwd Avg Packets/Bulk": 0,

    "Fwd Avg Bulk Rate": 0,

    "Bwd Avg Bytes/Bulk": 0,

    "Bwd Avg Packets/Bulk": 0,

    "Bwd Avg Bulk Rate": 0,

    "Subflow Fwd Packets": 89,

    "Subflow Fwd Bytes": 1398,

    "Subflow Bwd Packets": 113,

    "Subflow Bwd Bytes": 141366,

    "Init_Win_bytes_forward": 29200,

    "Init_Win_bytes_backward": 357,

    "act_data_pkt_fwd": 12,

    "min_seg_size_forward": 32,

    "Active Mean": 1030191.5,

    "Active Std": 1405456.045,

    "Active Max": 2023999,

    "Active Min": 36384,

    "Idle Mean": 58800000,

    "Idle Std": 214954.8046,

    "Idle Max": 59000000,

    "Idle Min": 58700000
}


# ============================================================
# 7. RUN PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("NETWORK TRAFFIC PREDICTION")
print("=" * 70)


result = predict_network_traffic(
    **sample
)


# ============================================================
# 8. DISPLAY RESULT
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


print("\nPrediction:")
print(
    f"  {prediction}"
)


print("\nConfidence:")
print(
    f"  {confidence:.2%}"
)


# ============================================================
# 9. DISPLAY ALL PROBABILITIES
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
# 10. FINAL DECISION
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
# 11. EXPECTED ORIGINAL LABEL
# ============================================================

print("\n")
print("=" * 70)
print("GROUND TRUTH")
print("=" * 70)

print(
    "Original dataset label: BENIGN"
)


# ============================================================
# 12. COMPARE PREDICTION WITH GROUND TRUTH
# ============================================================

actual_label = "BENIGN"


if prediction == actual_label:

    print(
        "Correct prediction: YES"
    )

else:

    print(
        "Correct prediction: NO"
    )

    print(
        f"Expected: {actual_label}"
    )

    print(
        f"Predicted: {prediction}"
    )


# ============================================================
# 13. FINAL SUMMARY
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
    f"Actual     : {actual_label}"
)

print(
    f"Features   : {len(feature_names)}"
)

print(
    "Model      : Random Forest"
)

print(
    f"Model path : {MODEL_PATH}"
)