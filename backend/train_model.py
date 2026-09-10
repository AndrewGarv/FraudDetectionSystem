import os

import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CSV_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_model.pt"
SCALER_PATH = "models/scaler.pkl"

BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 0.001


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
    torch.manual_seed(42)
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    # -----------------------------
    # Features and labels
    # -----------------------------
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # -----------------------------
    # Train / validation / test
    # -----------------------------
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

    # -----------------------------
    # Scale Time and Amount
    # -----------------------------
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[["Time", "Amount"]] = scaler.fit_transform(
        X_train[["Time", "Amount"]]
    )

    X_val[["Time", "Amount"]] = scaler.transform(
        X_val[["Time", "Amount"]]
    )

    X_test[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )

    # Save the fitted scaler for future inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print(f"Scaler saved to {SCALER_PATH}")


    # -----------------------------
    # Convert to PyTorch tensors
    # -----------------------------
    X_train_tensor = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).unsqueeze(1)

    X_val_tensor = torch.tensor(
        X_val.values,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val.values,
        dtype=torch.float32
    ).unsqueeze(1)

    # -----------------------------
    # Create datasets/loaders
    # -----------------------------
    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------
    # Model
    # -----------------------------
    model = FraudClassifier(
        input_size=X_train.shape[1]
    ).to(device)

    print(model)

    # -----------------------------
    # Class imbalance handling
    # -----------------------------
    fraud_count = y_train.sum()
    legitimate_count = len(y_train) - fraud_count

    pos_weight = torch.tensor(
        [legitimate_count / fraud_count],
        dtype=torch.float32
    ).to(device)

    print(f"Fraud weight: {pos_weight.item():.2f}")

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------
    # Training
    # -----------------------------
    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(features)

            loss = loss_function(
                logits,
                labels
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        # -------------------------
        # Validation loss
        # -------------------------
        model.eval()

        total_val_loss = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                logits = model(features)

                loss = loss_function(
                    logits,
                    labels
                )

                total_val_loss += loss.item()

        average_val_loss = (
            total_val_loss / len(val_loader)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Train Loss: {average_train_loss:.4f} "
            f"| Validation Loss: {average_val_loss:.4f}"
        )

    # -----------------------------
    # Save trained model
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()