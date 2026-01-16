import pickle
import os
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import preprocess_pipeline

def train_and_evaluate():
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(current_dir)             
    
    # Input Data Path
    csv_path = os.path.join(project_root, "Data", "Raw_data.csv")
    
    # Development Output Paths
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Deployment Output Paths
    deploy_dir = os.path.join(project_root, "deployment")
    os.makedirs(deploy_dir, exist_ok=True)
    
    # filenames
    vec_filename = "vectorizer.pkl"
    model_filename = "sentiment_model.pkl"

    start_time = time.time()
    
    # temporary vectorizer
    dev_vec_path = os.path.join(models_dir, vec_filename)
    
    X_train, X_test, y_train, y_test, vectorizer = preprocess_pipeline(
        csv_path=csv_path,
        text_col="review",
        label_col="sentiment",
        save_vec_path=dev_vec_path
    )
    
    print(f"Data processed in {time.time() - start_time:.2f} seconds.")

    # 3. TRAIN MODEL
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    
    # Save Dev Model
    dev_model_path = os.path.join(models_dir, model_filename)
    with open(dev_model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Development model saved to: {dev_model_path}")

    # 4. EVALUATE
    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # Save report to JSON for Streamlit
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
    with open(os.path.join(deploy_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=4)


    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(models_dir, "confusion_matrix.png"))
    print(f"Graph saved to models/confusion_matrix.png")
    # plt.show()

    print(f"CREATING DEPLOYMENT PACKAGE in '{deploy_dir}'")

    # 1. Save Model
    deploy_model_path = os.path.join(deploy_dir, model_filename)
    with open(deploy_model_path, "wb") as f:
        pickle.dump(model, f)

    # 2. Save Vectorizer
    deploy_vec_path = os.path.join(deploy_dir, vec_filename)
    with open(deploy_vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    metadata = {
        "accuracy": accuracy,
        "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "LogisticRegression",
        "description": "Sentiment Analysis model trained on Movie Reviews"
    }
    with open(os.path.join(deploy_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    train_and_evaluate()