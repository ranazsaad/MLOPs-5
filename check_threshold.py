import mlflow
import os
import sys

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# read run id from file
with open("model_info.txt") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
acc = run.data.metrics.get("accuracy", 0)

print("Accuracy:", acc)

if acc < 0.85:
    print("No, FAILED threshold")
    sys.exit(1)

print("yess, PASSED threshold")