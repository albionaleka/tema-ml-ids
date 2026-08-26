#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone

import json
import numpy as np
import pandas as pd
import joblib


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    "/home/user/Documents/project"
)

MODEL_DIR = (
    BASE_DIR / "models"
)

ARTIFACT_DIR = (
    BASE_DIR / "artifacts" / "random_forest"
)

LOG_DIR = (
    BASE_DIR / "logs"
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

LOG_FILE = (
    LOG_DIR /
    "ml_alerts.json"
)


# ============================================================
# 3. BENIGN LABELS
# ============================================================

BENIGN_LABELS = {
    "BENIGN",
    "Benign",
    "benign",
    "NORMAL",
    "Normal",
    "normal",
}


# ============================================================
# 4. LOAD MODEL AND ARTIFACTS
# ============================================================

print("=" * 70)
print("LOADING RANDOM FOREST MODEL")
print("=" * 70)

print(
    f"\nModel     : {MODEL_PATH}"
)

print(
    f"Features  : {FEATURES_PATH}"
)

print(
    f"Imputer   : {IMPUTER_PATH}"
)

print(
    f"Encoder   : {ENCODER_PATH}"
)

print(
    f"Alert log : {LOG_FILE}"
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not FEATURES_PATH.exists():

    raise FileNotFoundError(
        f"Feature list not found:\n"
        f"{FEATURES_PATH}"
    )


if not IMPUTER_PATH.exists():

    raise FileNotFoundError(
        f"Imputer not found:\n"
        f"{IMPUTER_PATH}"
    )


if not ENCODER_PATH.exists():

    raise FileNotFoundError(
        f"Label encoder not found:\n"
        f"{ENCODER_PATH}"
    )


model = joblib.load(
    MODEL_PATH
)

feature_names = list(
    joblib.load(FEATURES_PATH)
)

imputer = joblib.load(
    IMPUTER_PATH
)

encoder = joblib.load(
    ENCODER_PATH
)


print(
    "\nModel loaded successfully."
)

print(
    f"Number of expected features: "
    f"{len(feature_names)}"
)


# ============================================================
# 5. MODEL VALIDATION
# ============================================================

if hasattr(
    model,
    "n_features_in_"
):

    if int(
        model.n_features_in_
    ) != len(feature_names):

        raise ValueError(
            "\nMODEL FEATURE MISMATCH\n"
            f"Model expects "
            f"{model.n_features_in_} features, "
            f"but selected_features.pkl contains "
            f"{len(feature_names)}."
        )


if hasattr(
    imputer,
    "n_features_in_"
):

    if int(
        imputer.n_features_in_
    ) != len(feature_names):

        raise ValueError(
            "\nIMPUTER FEATURE MISMATCH\n"
            f"Imputer expects "
            f"{imputer.n_features_in_} features, "
            f"but selected_features.pkl contains "
            f"{len(feature_names)}."
        )


if hasattr(
    imputer,
    "feature_names_in_"
):

    imputer_features = list(
        imputer.feature_names_in_
    )

    if imputer_features != feature_names:

        print(
            "\nWARNING:"
        )

        print(
            "Imputer feature order differs "
            "from selected_features.pkl."
        )

        print(
            "\nselected_features.pkl:"
        )

        for index, feature in enumerate(
            feature_names
        ):

            print(
                f"  {index:03d}: {feature}"
            )

        print(
            "\nimputer.feature_names_in_:"
        )

        for index, feature in enumerate(
            imputer_features
        ):

            print(
                f"  {index:03d}: {feature}"
            )

        raise ValueError(
            "\nThe imputer and selected_features.pkl "
            "do not contain identical feature ordering."
        )


# ============================================================
# 6. DISPLAY CLASSES
# ============================================================

print("\n")
print("=" * 70)
print("MODEL CLASSES")
print("=" * 70)

for index, label in enumerate(
    encoder.classes_
):

    print(
        f"{index}: {label}"
    )


# ============================================================
# 7. DISPLAY FEATURES
# ============================================================

print("\n")
print("=" * 70)
print("MODEL FEATURES")
print("=" * 70)

for index, feature in enumerate(
    feature_names
):

    print(
        f"{index:03d}: {feature}"
    )


# ============================================================
# 8. PREDICTION FUNCTION
# ============================================================

def predict_network_traffic(
    **features
):

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

        print(
            "\nWARNING:"
        )

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
    # EXACT training feature order.
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
    # Final numeric validation
    # --------------------------------------------------------

    data_imputed = np.asarray(
        data_imputed,
        dtype=np.float64
    )


    if not np.all(
        np.isfinite(data_imputed)
    ):

        raise ValueError(
            "Imputer output contains "
            "NaN or infinite values."
        )


    # --------------------------------------------------------
    # Check feature count
    # --------------------------------------------------------

    if data_imputed.shape[1] != len(
        feature_names
    ):

        raise ValueError(
            "Feature vector mismatch: "
            f"{data_imputed.shape[1]} supplied, "
            f"{len(feature_names)} expected."
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

    try:

        prediction = encoder.inverse_transform(
            [prediction_encoded]
        )[0]

    except Exception:

        prediction = str(
            prediction_encoded
        )


    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        data_imputed
    )[0]


    # --------------------------------------------------------
    # Probability dictionary
    # --------------------------------------------------------

    probability_dict = {

        str(label): float(probability)

        for label, probability

        in zip(
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

        "prediction":
            str(prediction),

        "confidence":
            confidence,

        "probabilities":
            probability_dict
    }


# ============================================================
# 9. SAVE ML ALERT
# ============================================================

def save_ml_alert(
    prediction,
    confidence,
    probabilities,
    features
):

    prediction = str(
        prediction
    ).strip()


    # --------------------------------------------------------
    # BENIGN traffic is NOT saved.
    # --------------------------------------------------------

    if prediction.casefold() in {
        label.casefold()
        for label in BENIGN_LABELS
    }:

        print(
            "\n[LOG] BENIGN traffic "
            "was NOT written to ml_alerts.json."
        )

        return False


    # --------------------------------------------------------
    # Create log directory
    # --------------------------------------------------------

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Extract only useful alert/context fields
    #
    # These are NOT the complete ML feature vector.
    # The complete feature vector is still used internally
    # for prediction.
    # --------------------------------------------------------

    destination_port = features.get(
        "Destination Port",
        0
    )

    flow_duration = features.get(
        "Flow Duration",
        0
    )

    fwd_packets = features.get(
        "Total Fwd Packets",
        0
    )

    bwd_packets = features.get(
        "Total Backward Packets",
        0
    )

    fwd_bytes = features.get(
        "Total Length of Fwd Packets",
        0
    )

    bwd_bytes = features.get(
        "Total Length of Bwd Packets",
        0
    )

    total_packets = (
        fwd_packets
        + bwd_packets
    )

    total_bytes = (
        fwd_bytes
        + bwd_bytes
    )


    # --------------------------------------------------------
    # Create compact alert
    # --------------------------------------------------------

    alert = {

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "source":
            "random_forest_ids",

        "prediction_label":
            prediction,

        "confidence_score":
            round(
                float(confidence),
                6
            ),

        "severity":
            "malicious",

        "src_ip":
            features.get(
                "src_ip",
            ),

        "src_port":
            features.get(
                "src_port",
            ),

        "dst_ip":
            features.get(
                "dst_ip",
            ),

        "dst_port":
            destination_port,

        "flow_duration":
            flow_duration,

        "total_packets":
            total_packets,

        "fwd_packets":
            fwd_packets,

        "bwd_packets":
            bwd_packets,

        "fwd_bytes":
            fwd_bytes,

        "bwd_bytes":
            bwd_bytes,

        "total_bytes":
            total_bytes,

        "tcp_flags": {

            "syn":
                features.get(
                    "SYN Flag Count",
                    0
                ),

            "fin":
                features.get(
                    "FIN Flag Count",
                    0
                ),

            "rst":
                features.get(
                    "RST Flag Count",
                    0
                ),

            "psh":
                features.get(
                    "PSH Flag Count",
                    0
                ),

            "ack":
                features.get(
                    "ACK Flag Count",
                    0
                ),

            "urg":
                features.get(
                    "URG Flag Count",
                    0
                )
        }
    }


    # --------------------------------------------------------
    # Append JSON Lines record
    # --------------------------------------------------------

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                alert,
                ensure_ascii=False,
                separators=(",", ":")
            )
        )

        file.write("\n")


    print(
        "\n[ALERT] Malicious prediction saved."
    )

    print(
        f"[ALERT] File: {LOG_FILE}"
    )

    return True

# ============================================================
# 10. CIC-IDS2017 SAMPLE
# ============================================================

sample = {

    # --------------------------------------------------------
    # Alert/context metadata
    #
    # These fields are NOT model features.
    # They are only used when creating the Wazuh alert.
    # --------------------------------------------------------

    "src_ip":
        "192.168.1.50",

    "src_port":
        49152,

    "dst_ip":
        "192.168.1.10",

    "protocol":
        6,


    # --------------------------------------------------------
    # CIC-IDS2017 ML FEATURES
    # --------------------------------------------------------

    "Destination Port": 80,

    "Flow Duration": 11312786,

    "Total Fwd Packets": 5,

    "Total Backward Packets": 0,

    "Total Length of Fwd Packets": 30,

    "Total Length of Bwd Packets": 0,

    "Fwd Packet Length Max": 6,

    "Fwd Packet Length Min": 6,

    "Fwd Packet Length Mean": 6,

    "Fwd Packet Length Std": 0,

    "Bwd Packet Length Max": 0,

    "Bwd Packet Length Min": 0,

    "Bwd Packet Length Mean": 0,

    "Bwd Packet Length Std": 0,

    "Flow Bytes/s": 2.651866658,

    "Flow Packets/s": 0.441977776,

    "Flow IAT Mean": 2828196.5,

    "Flow IAT Std": 5655643.001,

    "Flow IAT Max": 11300000,

    "Flow IAT Min": 273,

    "Fwd IAT Total": 11300000,

    "Fwd IAT Mean": 2828196.5,

    "Fwd IAT Std": 5655643.001,

    "Fwd IAT Max": 11300000,

    "Fwd IAT Min": 273,

    "Bwd IAT Total": 0,

    "Bwd IAT Mean": 0,

    "Bwd IAT Std": 0,

    "Bwd IAT Max": 0,

    "Bwd IAT Min": 0,

    "Fwd PSH Flags": 0,

    "Bwd PSH Flags": 0,

    "Fwd URG Flags": 0,

    "Bwd URG Flags": 0,

    "Fwd Header Length": 100,

    "Bwd Header Length": 0,

    "Fwd Packets/s": 0.441977776,

    "Bwd Packets/s": 0,

    "Min Packet Length": 6,

    "Max Packet Length": 6,

    "Packet Length Mean": 6,

    "Packet Length Std": 0,

    "Packet Length Variance": 0,

    "FIN Flag Count": 0,

    "SYN Flag Count": 0,

    "RST Flag Count": 0,

    "PSH Flag Count": 0,

    "ACK Flag Count": 0,

    "URG Flag Count": 0,

    "CWE Flag Count": 0,

    "ECE Flag Count": 0,

    "Down/Up Ratio": 0,

    "Average Packet Size": 7.2,

    "Avg Fwd Segment Size": 6,

    "Avg Bwd Segment Size": 0,

    "Fwd Header Length.1": 100,

    "Fwd Avg Bytes/Bulk": 0,

    "Fwd Avg Packets/Bulk": 0,

    "Fwd Avg Bulk Rate": 0,

    "Bwd Avg Bytes/Bulk": 0,

    "Bwd Avg Packets/Bulk": 0,

    "Bwd Avg Bulk Rate": 0,

    "Subflow Fwd Packets": 5,

    "Subflow Fwd Bytes": 30,

    "Subflow Bwd Packets": 0,

    "Subflow Bwd Bytes": 0,

    "Init_Win_bytes_forward": 256,

    "Init_Win_bytes_backward": -1,

    "act_data_pkt_fwd": 4,

    "min_seg_size_forward": 20,

    "Active Mean": 1125,

    "Active Std": 0,

    "Active Max": 1125,

    "Active Min": 1125,

    "Idle Mean": 11300000,

    "Idle Std": 0,

    "Idle Max": 11300000,

    "Idle Min": 11300000
}

# ============================================================
# 11. RUN PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("NETWORK TRAFFIC PREDICTION")
print("=" * 70)


result = predict_network_traffic(
    **sample
)


# ============================================================
# 12. EXTRACT RESULT
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
# 13. DISPLAY PREDICTION
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
# 14. DISPLAY PROBABILITIES
# ============================================================

print("\nClass probabilities:")


sorted_probabilities = sorted(

    probabilities.items(),

    key=lambda item:
        item[1],

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
# 15. SAVE ONLY MALICIOUS PREDICTIONS
# ============================================================

alert_saved = save_ml_alert(

    prediction=prediction,

    confidence=confidence,

    probabilities=probabilities,

    features=sample
)


# ============================================================
# 16. FINAL DECISION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DECISION")
print("=" * 70)


if prediction.casefold() in {

    label.casefold()

    for label in BENIGN_LABELS
}:

    print(
        "Traffic classification: BENIGN"
    )

    print(
        "Alert saved: NO"
    )

else:

    print(
        "Traffic classification: ATTACK"
    )

    print(
        f"Attack category: {prediction}"
    )

    print(
        f"Alert saved: "
        f"{'YES' if alert_saved else 'NO'}"
    )


# ============================================================
# 17. RESULT SUMMARY
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
    "Model      : Random Forest"
)

print(
    f"Model path : {MODEL_PATH}"
)

print(
    f"Alert log  : {LOG_FILE}"
)

print(
    f"Alert saved: "
    f"{'YES' if alert_saved else 'NO'}"
)

print()