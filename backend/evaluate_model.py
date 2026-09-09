import pandas as pd
import torch
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
 
from torch import nn
 
 
CSV_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_model.pt"
 
 
class FraudClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
 
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
 
            nn.Linear(64, 32),
            nn.ReLU(),
 
            nn.Linear(32, 1)
        )
 
    def forward(self, x):
        return self.network(x)
 
 
def main():
    print("Loading dataset...")
 
    df = pd.read_csv(CSV_PATH)
 
    X = df.drop(columns=["Class"])
    y = df["Class"]
 
    # ---------------------------------
    # Recreate the same train/test split
    # ---------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
 
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
 
    # ---------------------------------
    # Recreate the same scaling
    # ---------------------------------
    scaler = StandardScaler()
 
    X_train = X_train.copy()
    X_test = X_test.copy()
 
    scaler.fit(X_train[["Time", "Amount"]])
 
    X_train[["Time", "Amount"]] = scaler.transform(
        X_train[["Time", "Amount"]]
    )
 
    X_test[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )
 
    # ---------------------------------
    # Convert test set to tensor
    # ---------------------------------
    X_test_tensor = torch.tensor(
        X_test.values,
        dtype=torch.float32
    )
 
    # ---------------------------------
    # Load trained model
    # ---------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
 
    model = FraudClassifier(
        input_size=X_test.shape[1]
    ).to(device)
 
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )
 
    model.eval()
 
    X_test_tensor = X_test_tensor.to(device)
 
    # ---------------------------------
    # Generate predictions
    # ---------------------------------
    with torch.no_grad():
 
        logits = model(X_test_tensor)
 
        probabilities = torch.sigmoid(logits)
 
    probabilities = (
        probabilities
        .cpu()
        .numpy()
        .flatten()
    )
 
    # ---------------------------------
    # Compare different thresholds
    # ---------------------------------
    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99
    ]
 
    print("\nTHRESHOLD COMPARISON")
    print("------------------------------------------------------------")
    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Flagged':<12}"
    )
    print("------------------------------------------------------------")
 
    for threshold in thresholds:
 
        threshold_predictions = (
            probabilities >= threshold
        ).astype(int)
 
        precision = precision_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )
 
        recall = recall_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )
 
        f1 = f1_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )
 
        flagged = threshold_predictions.sum()
 
        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{flagged:<12}"
        )
 
    # ---------------------------------
    # Use 0.50 for the main evaluation
    # ---------------------------------
    threshold = 0.50
 
    predictions = (
        probabilities >= threshold
    ).astype(int)
 
    # ---------------------------------
    # Metrics
    # ---------------------------------
    accuracy = accuracy_score(
        y_test,
        predictions
    )
 
    precision = precision_score(
        y_test,
        predictions
    )
 
    recall = recall_score(
        y_test,
        predictions
    )
 
    f1 = f1_score(
        y_test,
        predictions
    )
 
    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )
 
    print("\nMODEL PERFORMANCE")
    print("-----------------")
 
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
 
    # ---------------------------------
    # Confusion matrix
    # ---------------------------------
    cm = confusion_matrix(
        y_test,
        predictions
    )
 
    print("\nCONFUSION MATRIX")
    print("----------------")
    print(cm)
 
    tn, fp, fn, tp = cm.ravel()
 
    print("\nDetailed results:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
 
    print("\nCLASSIFICATION REPORT")
    print("---------------------")
 
    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )
 
 
if __name__ == "__main__":
    main()