#!/bin/bash
set -e

PROJECT_DIR="/home/user/Documents/project"
VENV_DIR="$PROJECT_DIR/venv"
SCRIPT="$PROJECT_DIR/scripts/live_ids.py"

LOG_DIR="$PROJECT_DIR/logs"
ALERT_FILE="$LOG_DIR/ml_alerts.json"

SERVICE_NAME="ml-ids.service"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME"

PYTHON="$VENV_DIR/bin/python"

echo "        CIC-IDS2017 REAL-TIME ML IDS DEPLOYMENT"

echo "[1/8] Verifying project..."

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory does not exist:"
    echo "$PROJECT_DIR"
    exit 1
fi


if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python virtual environment not found:"
    echo "$PYTHON"
    exit 1
fi


if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Analyzer script not found:"
    echo "$SCRIPT"
    exit 1
fi

echo "[2/8] Creating log directory..."

mkdir -p "$LOG_DIR"

echo "[3/8] Preparing ml_alerts.json..."

if [ ! -f "$ALERT_FILE" ]; then

    echo "[]" > "$ALERT_FILE"

fi

echo "[4/8] Setting permissions..."

chown -R root:root "$LOG_DIR"

chmod 755 "$LOG_DIR"

chmod 644 "$ALERT_FILE"

echo "[5/8] Checking Python dependencies..."

"$PYTHON" -c "
import scapy
import joblib
import sklearn
import numpy
print('Python dependencies OK')
"

echo "[6/8] Creating systemd service..."

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=CIC-IDS2017 Real-Time Machine Learning Intrusion Detection System
Documentation=Internal ML IDS
After=network-online.target
Wants=network-online.target

[Service]

Type=simple

User=root
Group=root

WorkingDirectory=$PROJECT_DIR

ExecStart=$PYTHON $SCRIPT

Restart=always
RestartSec=5s

Environment=PYTHONUNBUFFERED=1

StandardOutput=journal
StandardError=journal

# Give the service enough time to terminate cleanly.
TimeoutStopSec=10

# Security-related capabilities.
# Root already has these capabilities, but they explicitly document
# the privileges required by Scapy.
AmbientCapabilities=CAP_NET_RAW CAP_NET_ADMIN
CapabilityBoundingSet=CAP_NET_RAW CAP_NET_ADMIN

# Allow Scapy/network inspection.
NoNewPrivileges=false

[Install]
WantedBy=multi-user.target
EOF

echo "[7/8] Reloading systemd..."

systemctl daemon-reload

systemctl enable "$SERVICE_NAME"

echo "[8/8] Starting ML IDS..."

systemctl restart "$SERVICE_NAME"

echo ""
echo "             DEPLOYMENT COMPLETE"

echo ""
echo "Service:"
echo "  $SERVICE_NAME"

echo ""
echo "Status:"
echo "  systemctl status $SERVICE_NAME"

echo ""
echo "Live logs:"
echo "  journalctl -u $SERVICE_NAME -f"

echo ""
echo "ML alerts:"
echo "  $ALERT_FILE"

echo ""
echo "Project:"
echo "  $PROJECT_DIR"

echo ""