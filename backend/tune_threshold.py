import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
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

    # Same splits used during training
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

    # Scale using training data only
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val = X_val.copy()

    scaler.fit(X_train[["Time", "Amount"]])

    X_val[["Time", "Amount"]] = scaler.transform(
        X_val[["Time", "Amount"]]
    )

    # Convert validation data to tensor
    X_val_tensor = torch.tensor(
        X_val.values,
        dtype=torch.float32
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = FraudClassifier(
        input_size=X_val.shape[1]
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    X_val_tensor = X_val_tensor.to(device)

    # Generate validation scores
    with torch.no_grad():
        logits = model(X_val_tensor)
        probabilities = torch.sigmoid(logits)

    probabilities = (
        probabilities
        .cpu()
        .numpy()
        .flatten()
    )

    # Search thresholds from 0.500 to 0.999
    thresholds = np.arange(
        0.500,
        1.000,
        0.001
    )

    best_threshold = None
    best_f1 = -1
    best_precision = None
    best_recall = None

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    print("\nBEST VALIDATION THRESHOLD")
    print("-------------------------")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall:    {best_recall:.4f}")
    print(f"F1 Score:  {best_f1:.4f}")


if __name__ == "__main__":
    main()