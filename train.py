import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

mlflow.set_experiment("assignment5")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    mlflow.log_metric("accuracy", acc)
    
    # Save run ID to file directly from the script
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)
    
    print("Accuracy:", acc)
    print("RUN_ID:", run_id)