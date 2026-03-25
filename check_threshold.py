cat > check_threshold.py << 'EOF'
import mlflow
import os
import sys

# Set tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file://./mlruns"))

# Read run ID
try:
    with open("model_info.txt") as f:
        run_id = f.read().strip()
except FileNotFoundError:
    print("Error: model_info.txt not found")
    sys.exit(1)

# Get run from MLflow
try:
    run = mlflow.get_run(run_id)
    acc = run.data.metrics.get("accuracy", 0)
except Exception as e:
    print(f"Error fetching run: {e}")
    sys.exit(1)

print(f"Accuracy: {acc}")

if acc < 0.85:
    print("No, FAILED threshold")
    sys.exit(1)

print("Yes, PASSED threshold")
EOF