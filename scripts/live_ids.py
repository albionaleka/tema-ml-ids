#!/usr/bin/env python3

import os
import json
import time
import queue
import threading
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from scapy.all import (
    sniff,
    IP,
    TCP,
    UDP,
    ICMP,
)


# ============================================================
# CONFIGURATION
# ============================================================

INTERFACE = "enp0s17"

BASE_DIR = Path("/home/user/Documents/project")

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest.joblib"
)

FEATURES_PATH = (
    BASE_DIR
    / "artifacts"
    / "random_forest"
    / "selected_features.pkl"
)

IMPUTER_PATH = (
    BASE_DIR
    / "artifacts"
    / "random_forest"
    / "imputer.pkl"
)

LABEL_ENCODER_PATH = (
    BASE_DIR
    / "artifacts"
    / "random_forest"
    / "label_encoder.pkl"
)

LOG_DIR = BASE_DIR / "logs"

LOG_FILE = LOG_DIR / "ml_alerts.json"


# Flow inactivity timeout.
FLOW_TIMEOUT = 15.0

# Evaluate very large flows immediately.
MAX_FLOW_PACKETS = 100

# Cleanup interval.
CLEANUP_INTERVAL = 2.0

# CICFlowMeter active/idle threshold.
ACTIVE_IDLE_THRESHOLD = 1.0


BENIGN_LABELS = {
    "BENIGN",
    "Benign",
    "benign",
}


# ============================================================
# GLOBAL STATE
# ============================================================

flows = {}

flows_lock = threading.Lock()

prediction_queue = queue.Queue()

stop_event = threading.Event()


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("CIC-IDS2017 LIVE ML IDS")
print("=" * 70)

print(f"Interface : {INTERFACE}")
print(f"Model     : {MODEL_PATH}")
print(f"Features  : {FEATURES_PATH}")
print(f"Imputer   : {IMPUTER_PATH}")
print(f"Log file  : {LOG_FILE}")
print()


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"RandomForest model not found:\n{MODEL_PATH}"
    )

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"selected_features.pkl not found:\n{FEATURES_PATH}"
    )


model = joblib.load(MODEL_PATH)

selected_features = list(
    joblib.load(FEATURES_PATH)
)


print(
    f"Loaded model: {type(model).__name__}"
)

print(
    f"Expected features: {len(selected_features)}"
)


# ============================================================
# LOAD IMPUTER
# ============================================================

imputer = None

if IMPUTER_PATH.exists():

    try:

        imputer = joblib.load(
            IMPUTER_PATH
        )

        print("Loaded imputer.")

    except Exception as exc:

        print(
            f"WARNING: Could not load imputer: {exc}"
        )


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

label_encoder = None

if LABEL_ENCODER_PATH.exists():

    try:

        label_encoder = joblib.load(
            LABEL_ENCODER_PATH
        )

        print("Loaded label encoder.")

    except Exception as exc:

        print(
            f"WARNING: Could not load label encoder: {exc}"
        )


# ============================================================
# MODEL / PREPROCESSING VALIDATION
# ============================================================

if hasattr(model, "n_features_in_"):

    expected_model_features = int(
        model.n_features_in_
    )

    print(
        f"Model expects: "
        f"{expected_model_features} features"
    )

    if expected_model_features != len(
        selected_features
    ):

        raise ValueError(
            "\nFEATURE MISMATCH\n"
            f"Model expects "
            f"{expected_model_features} features, "
            f"but selected_features.pkl contains "
            f"{len(selected_features)}."
        )


if imputer is not None:

    if hasattr(imputer, "n_features_in_"):

        if (
            int(imputer.n_features_in_)
            != len(selected_features)
        ):

            raise ValueError(
                "\nIMPUTER FEATURE MISMATCH\n"
                f"Imputer expects "
                f"{imputer.n_features_in_} features, "
                f"but selected_features.pkl contains "
                f"{len(selected_features)}."
            )


    if hasattr(imputer, "feature_names_in_"):

        imputer_features = list(
            imputer.feature_names_in_
        )

        if imputer_features != selected_features:

            print(
                "\nWARNING: Imputer feature names/order "
                "differ from selected_features.pkl."
            )

            print(
                "\nselected_features.pkl:"
            )

            for i, feature in enumerate(
                selected_features
            ):

                print(
                    f"  {i:03d}: {feature}"
                )

            print(
                "\nimputer.feature_names_in_:"
            )

            for i, feature in enumerate(
                imputer_features
            ):

                print(
                    f"  {i:03d}: {feature}"
                )

            raise ValueError(
                "\nThe imputer and selected_features.pkl "
                "do not contain identical feature ordering."
            )


# ============================================================
# PRINT FEATURES
# ============================================================

print("\nSelected feature order:")

for index, feature in enumerate(
    selected_features
):

    print(
        f"{index:03d}: {feature}"
    )

print()


# ============================================================
# FLOW CLASS
# ============================================================

class Flow:

    def __init__(
        self,
        src_ip,
        src_port,
        dst_ip,
        dst_port,
        protocol
    ):

        # Original direction of the flow.
        self.src_ip = src_ip
        self.src_port = src_port

        self.dst_ip = dst_ip
        self.dst_port = dst_port

        self.protocol = protocol

        self.start_time = None
        self.last_seen = None

        self.packet_count = 0

        self.forward_packets = []
        self.backward_packets = []

        self.forward_bytes = 0
        self.backward_bytes = 0

        self.forward_packet_lengths = []
        self.backward_packet_lengths = []

        self.forward_iats = []
        self.backward_iats = []

        self.all_timestamps = []

        # TCP flags.
        self.syn_count = 0
        self.fin_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.ack_count = 0
        self.urg_count = 0

        # ECE/CWR.
        self.ece_count = 0
        self.cwr_count = 0

        # Header bytes.
        self.forward_header_bytes = 0
        self.backward_header_bytes = 0

        # Last packet timestamps by direction.
        self.forward_last_timestamp = None
        self.backward_last_timestamp = None

        # Active/idle periods.
        self.active_periods = []
        self.idle_periods = []

        self.current_active_start = None
        self.current_active_end = None

        self.last_packet_timestamp = None


# ============================================================
# FLOW KEY
# ============================================================

def canonical_flow_key(
    src_ip,
    src_port,
    dst_ip,
    dst_port,
    protocol
):

    endpoint_a = (
        str(src_ip),
        int(src_port),
    )

    endpoint_b = (
        str(dst_ip),
        int(dst_port),
    )

    if endpoint_a <= endpoint_b:

        first = endpoint_a
        second = endpoint_b

    else:

        first = endpoint_b
        second = endpoint_a

    return (
        first[0],
        first[1],
        second[0],
        second[1],
        int(protocol)
    )


# ============================================================
# PACKET INFORMATION
# ============================================================

def extract_packet_info(packet):

    if not packet.haslayer(IP):

        return None

    ip = packet[IP]

    src_ip = str(ip.src)
    dst_ip = str(ip.dst)

    protocol = int(ip.proto)

    src_port = 0
    dst_port = 0

    tcp_flags = set()

    ip_header_length = 20

    if getattr(ip, "ihl", None):

        try:

            ip_header_length = int(
                ip.ihl
            ) * 4

        except Exception:

            ip_header_length = 20


    transport_header_length = 0


    # --------------------------------------------------------
    # TCP
    # --------------------------------------------------------

    if packet.haslayer(TCP):

        tcp = packet[TCP]

        src_port = int(tcp.sport)
        dst_port = int(tcp.dport)

        flags = int(tcp.flags)

        if flags & 0x02:
            tcp_flags.add("SYN")

        if flags & 0x01:
            tcp_flags.add("FIN")

        if flags & 0x04:
            tcp_flags.add("RST")

        if flags & 0x08:
            tcp_flags.add("PSH")

        if flags & 0x10:
            tcp_flags.add("ACK")

        if flags & 0x20:
            tcp_flags.add("URG")

        # TCP ECE.
        if flags & 0x40:
            tcp_flags.add("ECE")

        # TCP CWR.
        if flags & 0x80:
            tcp_flags.add("CWR")

        try:

            transport_header_length = (
                int(tcp.dataofs) * 4
            )

        except Exception:

            transport_header_length = 20


    # --------------------------------------------------------
    # UDP
    # --------------------------------------------------------

    elif packet.haslayer(UDP):

        udp = packet[UDP]

        src_port = int(udp.sport)
        dst_port = int(udp.dport)

        transport_header_length = 8


    # --------------------------------------------------------
    # ICMP
    # --------------------------------------------------------

    elif packet.haslayer(ICMP):

        src_port = 0
        dst_port = 0

        transport_header_length = 8


    else:

        return None


    packet_length = len(packet)

    timestamp = float(
        getattr(
            packet,
            "time",
            time.time()
        )
    )


    total_header_length = (
        ip_header_length
        + transport_header_length
    )


    return {

        "src_ip": src_ip,

        "src_port": src_port,

        "dst_ip": dst_ip,

        "dst_port": dst_port,

        "protocol": protocol,

        "length": packet_length,

        "timestamp": timestamp,

        "tcp_flags": tcp_flags,

        "ip_header_length":
            ip_header_length,

        "transport_header_length":
            transport_header_length,

        "header_length":
            total_header_length,
    }


# ============================================================
# ACTIVE / IDLE TRACKING
# ============================================================

def update_active_idle(
    flow,
    timestamp
):

    if flow.last_packet_timestamp is None:

        flow.current_active_start = timestamp
        flow.current_active_end = timestamp

        flow.last_packet_timestamp = timestamp

        return


    gap = (
        timestamp
        - flow.last_packet_timestamp
    )

    # Large gap => end active period and begin idle.
    if gap > ACTIVE_IDLE_THRESHOLD:

        if (
            flow.current_active_start
            is not None
            and flow.current_active_end
            is not None
        ):

            active_duration = (
                flow.current_active_end
                - flow.current_active_start
            )

            flow.active_periods.append(
                max(active_duration, 0.0)
            )


        idle_duration = (
            gap
        )

        flow.idle_periods.append(
            max(idle_duration, 0.0)
        )

        flow.current_active_start = timestamp
        flow.current_active_end = timestamp

    else:

        if flow.current_active_start is None:

            flow.current_active_start = timestamp

        flow.current_active_end = timestamp


    flow.last_packet_timestamp = timestamp


def finalize_active_idle(flow):

    if (
        flow.current_active_start
        is not None
        and flow.current_active_end
        is not None
    ):

        duration = (
            flow.current_active_end
            - flow.current_active_start
        )

        flow.active_periods.append(
            max(duration, 0.0)
        )

    flow.current_active_start = None
    flow.current_active_end = None


# ============================================================
# UPDATE FLOW
# ============================================================

def update_flow(
    flow,
    packet_info,
    forward=True
):

    timestamp = packet_info["timestamp"]

    length = packet_info["length"]

    if flow.start_time is None:

        flow.start_time = timestamp

    flow.last_seen = timestamp

    flow.packet_count += 1

    flow.all_timestamps.append(
        timestamp
    )

    update_active_idle(
        flow,
        timestamp
    )


    flags = packet_info["tcp_flags"]


    if "SYN" in flags:
        flow.syn_count += 1

    if "FIN" in flags:
        flow.fin_count += 1

    if "RST" in flags:
        flow.rst_count += 1

    if "PSH" in flags:
        flow.psh_count += 1

    if "ACK" in flags:
        flow.ack_count += 1

    if "URG" in flags:
        flow.urg_count += 1

    if "ECE" in flags:
        flow.ece_count += 1

    if "CWR" in flags:
        flow.cwr_count += 1


    header_length = (
        packet_info["header_length"]
    )


    if forward:

        flow.forward_packets.append(
            packet_info
        )

        flow.forward_bytes += length

        flow.forward_packet_lengths.append(
            length
        )

        if (
            flow.forward_last_timestamp
            is not None
        ):

            flow.forward_iats.append(
                timestamp
                - flow.forward_last_timestamp
            )

        flow.forward_last_timestamp = (
            timestamp
        )

        flow.forward_header_bytes += (
            header_length
        )


    else:

        flow.backward_packets.append(
            packet_info
        )

        flow.backward_bytes += length

        flow.backward_packet_lengths.append(
            length
        )

        if (
            flow.backward_last_timestamp
            is not None
        ):

            flow.backward_iats.append(
                timestamp
                - flow.backward_last_timestamp
            )

        flow.backward_last_timestamp = (
            timestamp
        )

        flow.backward_header_bytes += (
            header_length
        )


# ============================================================
# SAFE STATISTICS
# ============================================================

def safe_sum(values):

    if not values:
        return 0.0

    return float(np.sum(values))


def safe_mean(values):

    if not values:
        return 0.0

    return float(np.mean(values))


def safe_std(values):

    if len(values) < 2:
        return 0.0

    return float(np.std(values))


def safe_min(values):

    if not values:
        return 0.0

    return float(np.min(values))


def safe_max(values):

    if not values:
        return 0.0

    return float(np.max(values))


def safe_median(values):

    if not values:
        return 0.0

    return float(np.median(values))


def safe_variance(values):

    if len(values) < 2:
        return 0.0

    return float(np.var(values))


# ============================================================
# BULK FEATURES
# ============================================================

def calculate_bulk_features(
    packets,
    direction
):
    """
    Calculate approximate CICFlowMeter-style bulk statistics.

    A bulk is considered to have started after at least four
    packets have accumulated in the same direction without a
    large idle gap.

    Returns:

        avg_bytes_bulk
        avg_packets_bulk
        avg_bulk_rate
    """

    if len(packets) < 4:

        return (
            0.0,
            0.0,
            0.0
        )


    packet_lengths = [
        int(p["length"])
        for p in packets
    ]

    timestamps = [
        float(p["timestamp"])
        for p in packets
    ]


    bulks = []

    current_lengths = []
    current_start = None
    previous_time = None


    for length, timestamp in zip(
        packet_lengths,
        timestamps
    ):

        if previous_time is None:

            current_lengths = [length]
            current_start = timestamp

            previous_time = timestamp

            continue


        gap = (
            timestamp
            - previous_time
        )


        if gap > ACTIVE_IDLE_THRESHOLD:

            if len(current_lengths) >= 4:

                bulks.append(
                    (
                        current_lengths,
                        current_start,
                        previous_time
                    )
                )


            current_lengths = [length]
            current_start = timestamp

        else:

            current_lengths.append(
                length
            )


        previous_time = timestamp


    if len(current_lengths) >= 4:

        bulks.append(
            (
                current_lengths,
                current_start,
                previous_time
            )
        )


    if not bulks:

        return (
            0.0,
            0.0,
            0.0
        )


    bytes_per_bulk = []

    packets_per_bulk = []

    rates = []


    for (
        lengths,
        start,
        end
    ) in bulks:

        total_bytes = float(
            sum(lengths)
        )

        packet_count = float(
            len(lengths)
        )

        duration = (
            end - start
        )

        if duration > 0:

            rate = (
                total_bytes
                / duration
            )

        else:

            rate = 0.0


        bytes_per_bulk.append(
            total_bytes
        )

        packets_per_bulk.append(
            packet_count
        )

        rates.append(
            rate
        )


    return (

        safe_mean(
            bytes_per_bulk
        ),

        safe_mean(
            packets_per_bulk
        ),

        safe_mean(
            rates
        )
    )


# ============================================================
# FLOW FEATURE EXTRACTION
# ============================================================

def extract_features(flow):

    finalize_active_idle(
        flow
    )


    if flow.start_time is None:

        duration = 0.0

    else:

        duration = max(
            flow.last_seen
            - flow.start_time,
            0.0
        )


    duration_us = (
        duration
        * 1_000_000.0
    )


    fwd_lengths = (
        flow.forward_packet_lengths
    )

    bwd_lengths = (
        flow.backward_packet_lengths
    )

    all_lengths = (
        fwd_lengths
        + bwd_lengths
    )


    fwd_iats = (
        flow.forward_iats
    )

    bwd_iats = (
        flow.backward_iats
    )

    all_iats = (
        fwd_iats
        + bwd_iats
    )


    total_packets = (
        len(fwd_lengths)
        + len(bwd_lengths)
    )


    total_bytes = (
        flow.forward_bytes
        + flow.backward_bytes
    )


    # --------------------------------------------------------
    # Bulk
    # --------------------------------------------------------

    (
        fwd_avg_bytes_bulk,
        fwd_avg_packets_bulk,
        fwd_avg_bulk_rate
    ) = calculate_bulk_features(
        flow.forward_packets,
        "forward"
    )


    (
        bwd_avg_bytes_bulk,
        bwd_avg_packets_bulk,
        bwd_avg_bulk_rate
    ) = calculate_bulk_features(
        flow.backward_packets,
        "backward"
    )


    # --------------------------------------------------------
    # Active / idle
    # --------------------------------------------------------

    active = flow.active_periods

    idle = flow.idle_periods


    active_mean = (
        safe_mean(active)
        * 1_000_000.0
    )

    active_std = (
        safe_std(active)
        * 1_000_000.0
    )

    active_max = (
        safe_max(active)
        * 1_000_000.0
    )

    active_min = (
        safe_min(active)
        * 1_000_000.0
    )


    idle_mean = (
        safe_mean(idle)
        * 1_000_000.0
    )

    idle_std = (
        safe_std(idle)
        * 1_000_000.0
    )

    idle_max = (
        safe_max(idle)
        * 1_000_000.0
    )

    idle_min = (
        safe_min(idle)
        * 1_000_000.0
    )


    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    feature_values = {

        "Destination Port":
            flow.dst_port,

        "Flow Duration":
            duration_us,

        "Total Fwd Packets":
            len(fwd_lengths),

        "Total Backward Packets":
            len(bwd_lengths),

        "Total Length of Fwd Packets":
            flow.forward_bytes,

        "Total Length of Bwd Packets":
            flow.backward_bytes,

        "Fwd Packet Length Max":
            safe_max(fwd_lengths),

        "Fwd Packet Length Min":
            safe_min(fwd_lengths),

        "Fwd Packet Length Mean":
            safe_mean(fwd_lengths),

        "Fwd Packet Length Std":
            safe_std(fwd_lengths),

        "Bwd Packet Length Max":
            safe_max(bwd_lengths),

        "Bwd Packet Length Min":
            safe_min(bwd_lengths),

        "Bwd Packet Length Mean":
            safe_mean(bwd_lengths),

        "Bwd Packet Length Std":
            safe_std(bwd_lengths),

        "Flow Bytes/s":
            (
                total_bytes / duration
                if duration > 0
                else 0.0
            ),

        "Flow Packets/s":
            (
                total_packets / duration
                if duration > 0
                else 0.0
            ),

        "Flow IAT Mean":
            safe_mean(all_iats),

        "Flow IAT Std":
            safe_std(all_iats),

        "Flow IAT Max":
            safe_max(all_iats),

        "Flow IAT Min":
            safe_min(all_iats),

        "Fwd IAT Total":
            safe_sum(fwd_iats),

        "Fwd IAT Mean":
            safe_mean(fwd_iats),

        "Fwd IAT Std":
            safe_std(fwd_iats),

        "Fwd IAT Max":
            safe_max(fwd_iats),

        "Fwd IAT Min":
            safe_min(fwd_iats),

        "Bwd IAT Total":
            safe_sum(bwd_iats),

        "Bwd IAT Mean":
            safe_mean(bwd_iats),

        "Bwd IAT Std":
            safe_std(bwd_iats),

        "Bwd IAT Max":
            safe_max(bwd_iats),

        "Bwd IAT Min":
            safe_min(bwd_iats),

        "Fwd PSH Flags":
            sum(
                1
                for p in flow.forward_packets
                if "PSH" in p["tcp_flags"]
            ),

        "Bwd PSH Flags":
            sum(
                1
                for p in flow.backward_packets
                if "PSH" in p["tcp_flags"]
            ),

        "Fwd URG Flags":
            sum(
                1
                for p in flow.forward_packets
                if "URG" in p["tcp_flags"]
            ),

        "Bwd URG Flags":
            sum(
                1
                for p in flow.backward_packets
                if "URG" in p["tcp_flags"]
            ),

        "Fwd Header Length":
            flow.forward_header_bytes,

        # Some CIC-IDS2017 CSV variants contain a duplicate
        # Fwd Header Length column named Fwd Header Length.1.
        "Fwd Header Length.1":
            flow.forward_header_bytes,

        "Bwd Header Length":
            flow.backward_header_bytes,

        "Fwd Packets/s":
            (
                len(fwd_lengths) / duration
                if duration > 0
                else 0.0
            ),

        "Bwd Packets/s":
            (
                len(bwd_lengths) / duration
                if duration > 0
                else 0.0
            ),

        "Min Packet Length":
            safe_min(all_lengths),

        "Max Packet Length":
            safe_max(all_lengths),

        "Packet Length Mean":
            safe_mean(all_lengths),

        "Packet Length Std":
            safe_std(all_lengths),

        "Packet Length Variance":
            safe_variance(all_lengths),

        "FIN Flag Count":
            flow.fin_count,

        "SYN Flag Count":
            flow.syn_count,

        "RST Flag Count":
            flow.rst_count,

        "PSH Flag Count":
            flow.psh_count,

        "ACK Flag Count":
            flow.ack_count,

        "URG Flag Count":
            flow.urg_count,

        # CIC-IDS2017 terminology.
        "CWE Flag Count":
            flow.cwr_count,

        "ECE Flag Count":
            flow.ece_count,

        "Down/Up Ratio":
            (
                len(bwd_lengths)
                / len(fwd_lengths)
                if len(fwd_lengths) > 0
                else 0.0
            ),

        "Average Packet Size":
            (
                total_bytes / total_packets
                if total_packets > 0
                else 0.0
            ),

        "Avg Fwd Segment Size":
            safe_mean(fwd_lengths),

        "Avg Bwd Segment Size":
            safe_mean(bwd_lengths),

        "Fwd Avg Bytes/Bulk":
            fwd_avg_bytes_bulk,

        "Fwd Avg Packets/Bulk":
            fwd_avg_packets_bulk,

        "Fwd Avg Bulk Rate":
            fwd_avg_bulk_rate,

        "Bwd Avg Bytes/Bulk":
            bwd_avg_bytes_bulk,

        "Bwd Avg Packets/Bulk":
            bwd_avg_packets_bulk,

        "Bwd Avg Bulk Rate":
            bwd_avg_bulk_rate,

        "Subflow Fwd Packets":
            len(fwd_lengths),

        "Subflow Fwd Bytes":
            flow.forward_bytes,

        "Subflow Bwd Packets":
            len(bwd_lengths),

        "Subflow Bwd Bytes":
            flow.backward_bytes,

        "Init_Win_bytes_forward":
            0,

        "Init_Win_bytes_backward":
            0,

        "act_data_pkt_fwd":
            sum(
                1
                for p in flow.forward_packets
                if p["length"] > p["header_length"]
            ),

        "min_seg_size_forward":
            safe_min(fwd_lengths),

        "Active Mean":
            active_mean,

        "Active Std":
            active_std,

        "Active Max":
            active_max,

        "Active Min":
            active_min,

        "Idle Mean":
            idle_mean,

        "Idle Std":
            idle_std,

        "Idle Max":
            idle_max,

        "Idle Min":
            idle_min,
    }


    # --------------------------------------------------------
    # Extra TCP-derived values
    # --------------------------------------------------------

    # Initial TCP windows.
    if flow.forward_packets:

        first_forward = (
            flow.forward_packets[0]
        )

        first_forward_packet = (
            first_forward
        )

        # Window isn't currently stored by extract_packet_info,
        # so this remains zero unless explicitly captured.
        feature_values[
            "Init_Win_bytes_forward"
        ] = first_forward_packet.get(
            "tcp_window",
            0
        )


    if flow.backward_packets:

        first_backward = (
            flow.backward_packets[0]
        )

        feature_values[
            "Init_Win_bytes_backward"
        ] = first_backward.get(
            "tcp_window",
            0
        )


    return feature_values


# ============================================================
# BUILD MODEL VECTOR
# ============================================================

def build_model_vector(flow):

    feature_values = extract_features(
        flow
    )


    vector = []

    missing_features = []


    for feature in selected_features:

        if feature in feature_values:

            value = feature_values[
                feature
            ]

        else:

            missing_features.append(
                feature
            )

            value = np.nan


        vector.append(value)


    # --------------------------------------------------------
    # Missing feature diagnostic
    # --------------------------------------------------------

    if missing_features:

        print(
            "\nWARNING: Missing live features:"
        )

        for feature in missing_features:

            print(
                f"  - {feature}"
            )

    else:

        print(
            "[FEATURES] All selected live "
            "features available."
        )


    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    vector = np.asarray(
        vector,
        dtype=np.float64
    )


    vector[
        ~np.isfinite(vector)
    ] = np.nan


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The imputer was fitted with feature names.
    # Therefore transform a DataFrame rather than a NumPy array.
    # This eliminates:
    #
    # "X does not have valid feature names..."
    # --------------------------------------------------------

    if imputer is not None:

        X_live = pd.DataFrame(
            [vector],
            columns=selected_features
        )

        X_live = imputer.transform(
            X_live
        )

        vector = X_live[0]


    else:

        vector = np.nan_to_num(
            vector,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if len(vector) != len(
        selected_features
    ):

        raise ValueError(
            "Feature vector mismatch: "
            f"{len(vector)} supplied, "
            f"{len(selected_features)} expected."
        )


    if hasattr(
        model,
        "n_features_in_"
    ):

        if len(vector) != (
            model.n_features_in_
        ):

            raise ValueError(
                "Model feature mismatch: "
                f"{len(vector)} supplied, "
                f"{model.n_features_in_} expected."
            )


    if not np.all(
        np.isfinite(vector)
    ):

        raise ValueError(
            "Feature vector still contains "
            "NaN or infinite values after preprocessing."
        )


    return vector


# ============================================================
# PREDICTION
# ============================================================

def predict_flow(flow):

    vector = build_model_vector(flow)

    # RandomForest was trained without feature names,
    # so inference must also use an unnamed NumPy array.
    X = np.asarray(
        vector,
        dtype=np.float64
    ).reshape(1, -1)

    prediction = model.predict(
        X
    )[0]

    probabilities = model.predict_proba(
        X
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    # --------------------------------------------------------
    # Decode prediction if required.
    # --------------------------------------------------------

    if (
        label_encoder is not None
        and isinstance(
            prediction,
            (int, np.integer)
        )
    ):

        try:

            prediction_label = (
                label_encoder
                .inverse_transform(
                    [prediction]
                )[0]
            )

        except Exception:

            prediction_label = str(
                prediction
            )

    else:

        prediction_label = str(
            prediction
        )

    return (
        prediction_label,
        confidence
    )

# ============================================================
# JSON ALERT WRITER
# ============================================================

class AlertWriter:
    """
    Background JSON Lines writer.

    Only malicious ML predictions are sent to this writer.
    """

    def __init__(self, output_file):

        self.output_file = Path(output_file)

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Create the file if it doesn't exist.
        self.output_file.touch(
            exist_ok=True
        )

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True
        )

    def start(self):

        self.thread.start()

    def stop(self):

        prediction_queue.put(None)

        self.thread.join(
            timeout=5
        )

    def _worker(self):

        while True:

            item = prediction_queue.get()

            try:

                if item is None:

                    return

                line = json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(",", ":")
                )

                with open(
                    self.output_file,
                    "a",
                    encoding="utf-8"
                ) as f:

                    f.write(
                        line + "\n"
                    )

                    f.flush()

                    os.fsync(
                        f.fileno()
                    )

            finally:

                prediction_queue.task_done()


alert_writer = AlertWriter(
    LOG_FILE
)

alert_writer.start()


# ============================================================
# CREATE ALERT
# ============================================================

def create_alert(
    flow,
    prediction_label,
    confidence
):

    label = str(prediction_label).strip()

    # Collect TCP flags seen in this flow.
    tcp_flags = []

    if flow.syn_count > 0:
        tcp_flags.append("SYN")

    if flow.fin_count > 0:
        tcp_flags.append("FIN")

    if flow.rst_count > 0:
        tcp_flags.append("RST")

    if flow.psh_count > 0:
        tcp_flags.append("PSH")

    if flow.ack_count > 0:
        tcp_flags.append("ACK")

    if flow.urg_count > 0:
        tcp_flags.append("URG")

    if flow.ece_count > 0:
        tcp_flags.append("ECE")

    if flow.cwr_count > 0:
        tcp_flags.append("CWR")

    # Flow duration in milliseconds.
    duration_ms = 0.0

    if (
        flow.start_time is not None
        and flow.last_seen is not None
    ):
        duration_ms = max(
            (
                flow.last_seen
                - flow.start_time
            ) * 1000.0,
            0.0
        )

    return {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "source": "random_forest_ids",

        "prediction_label": label,

        "confidence_score": round(
            float(confidence),
            6
        ),

        "severity": (
            "benign"
            if label.casefold() in BENIGN_LABELS
            else "malicious"
        ),

        "src_ip": flow.src_ip,
        "src_port": flow.src_port,

        "dst_ip": flow.dst_ip,
        "dst_port": flow.dst_port,

        "protocol": flow.protocol,

        "flow_duration": round(
            duration_ms,
            3
        ),

        "total_packets": flow.packet_count,

        "fwd_packets": len(
            flow.forward_packets
        ),

        "bwd_packets": len(
            flow.backward_packets
        ),

        "fwd_bytes": flow.forward_bytes,

        "bwd_bytes": flow.backward_bytes,

        "total_bytes": (
            flow.forward_bytes
            + flow.backward_bytes
        ),

        "tcp_flags": tcp_flags
    }

# ============================================================
# PROCESS COMPLETED FLOW
# ============================================================

# ============================================================
# PROCESS COMPLETED FLOW
# ============================================================

def process_flow(flow):

    try:

        prediction_label, confidence = predict_flow(flow)

        label = str(prediction_label).strip()

        print(
            f"[ML] "
            f"{flow.src_ip}:{flow.src_port} -> "
            f"{flow.dst_ip}:{flow.dst_port} "
            f"protocol={flow.protocol} "
            f"prediction={label} "
            f"confidence={confidence:.4f}"
        )

        if label.casefold() in {
            "benign",
            "normal"
        }:

            print(
                "[ML] Normal/benign traffic - "
                "not written to ml_alerts.json"
            )

            return

        event = create_alert(
            flow,
            label,
            confidence
        )

        prediction_queue.put(event)

        print(
            "[ALERT] "
            + json.dumps(
                event,
                ensure_ascii=False
            )
        )

    except Exception as exc:

        print(
            f"[ERROR] Flow prediction failed: {exc}"
        )

# ============================================================
# TAKE FLOW
# ============================================================

def take_flow(key):

    """
    Atomically remove a flow from the global dictionary.

    This prevents the packet handler and cleanup thread from
    both processing the same flow.
    """

    with flows_lock:

        return flows.pop(
            key,
            None
        )


# ============================================================
# PACKET HANDLER
# ============================================================

def packet_handler(packet):

    try:

        info = extract_packet_info(
            packet
        )


        if info is None:

            return


        key = canonical_flow_key(

            info["src_ip"],

            info["src_port"],

            info["dst_ip"],

            info["dst_port"],

            info["protocol"]
        )


        flow_to_process = None


        with flows_lock:

            flow = flows.get(
                key
            )


            if flow is None:

                # First packet determines forward direction.
                flow = Flow(

                    info["src_ip"],

                    info["src_port"],

                    info["dst_ip"],

                    info["dst_port"],

                    info["protocol"]
                )


                flows[key] = flow

                forward = True


            else:

                forward = (

                    info["src_ip"]
                    == flow.src_ip

                    and

                    info["src_port"]
                    == flow.src_port

                    and

                    info["dst_ip"]
                    == flow.dst_ip

                    and

                    info["dst_port"]
                    == flow.dst_port
                )


            update_flow(
                flow,
                info,
                forward
            )


            packet_count = (
                flow.packet_count
            )


            # Remove immediately under the lock.
            if packet_count >= MAX_FLOW_PACKETS:

                flow_to_process = flows.pop(
                    key,
                    None
                )


        # Process outside the lock.
        if flow_to_process is not None:

            process_flow(
                flow_to_process
            )


    except Exception as exc:

        print(
            f"[ERROR] Packet handler: {exc}"
        )


# ============================================================
# FLOW CLEANUP THREAD
# ============================================================

def flow_cleanup_worker():

    while not stop_event.is_set():

        stop_event.wait(
            CLEANUP_INTERVAL
        )


        if stop_event.is_set():

            break


        now = time.time()


        expired = []


        with flows_lock:

            for key, flow in list(
                flows.items()
            ):

                if flow.last_seen is None:

                    continue


                if (
                    now
                    - flow.last_seen
                    >= FLOW_TIMEOUT
                ):

                    # Remove under lock immediately.
                    removed = flows.pop(
                        key,
                        None
                    )


                    if removed is not None:

                        expired.append(
                            removed
                        )


        # Process outside lock.
        for flow in expired:

            print(
                f"[FLOW] Closing flow "
                f"{flow.src_ip}:{flow.src_port} -> "
                f"{flow.dst_ip}:{flow.dst_port}"
            )


            process_flow(
                flow
            )


cleanup_thread = threading.Thread(
    target=flow_cleanup_worker,
    daemon=True
)

cleanup_thread.start()


# ============================================================
# START CAPTURE
# ============================================================

print()
print("=" * 70)
print("STARTING LIVE PACKET CAPTURE")
print("=" * 70)

print(
    f"Interface: {INTERFACE}"
)

print(
    f"Flow timeout: {FLOW_TIMEOUT}s"
)

print(
    f"Max flow packets: {MAX_FLOW_PACKETS}"
)

print(
    "Press CTRL+C to stop."
)

print()


try:

    sniff(

        iface=INTERFACE,

        prn=packet_handler,

        store=False
    )


except KeyboardInterrupt:

    print(
        "\nStopping IDS..."
    )


except Exception as exc:

    print(
        f"\nSniffer error: {exc}"
    )


finally:

    stop_event.set()


    # --------------------------------------------------------
    # Atomically take all remaining flows.
    # --------------------------------------------------------

    with flows_lock:

        remaining = list(
            flows.values()
        )

        flows.clear()


    # --------------------------------------------------------
    # Process remaining flows.
    # --------------------------------------------------------

    for flow in remaining:

        print(
            f"[FLOW] Closing remaining flow "
            f"{flow.src_ip}:{flow.src_port} -> "
            f"{flow.dst_ip}:{flow.dst_port}"
        )


        process_flow(
            flow
        )


    # --------------------------------------------------------
    # Wait for queued alerts.
    # --------------------------------------------------------

    prediction_queue.join()


    # --------------------------------------------------------
    # Stop writer.
    # --------------------------------------------------------

    alert_writer.stop()


    print(
        "IDS stopped cleanly."
    )