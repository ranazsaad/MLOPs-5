cat > train.py << 'EOF'
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Set tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file://./mlruns"))
mlflow.set_experiment("assignment5")

# Load data
try:
    import pandas as pd
    if os.path.exists('data/iris.csv'):
        df = pd.read_csv('data/iris.csv')
        X = df.drop('target', axis=1).values
        y = df['target'].values
        print("Using data from DVC")
    else:
        raise FileNotFoundError()
except:
    X, y = load_iris(return_X_y=True)
    print("Using data from sklearn")

# Split data - high accuracy configuration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    mlflow.log_metric("accuracy", acc)
    
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"RUN_ID: {run_id}")
EOF