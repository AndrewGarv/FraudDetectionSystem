import joblib
import numpy as np
import torch
from torch import nn


MODEL_PATH = "models/fraud_model.pt"
SCALER_PATH = "models/scaler.pkl"

THRESHOLD = 0.997
INPUT_SIZE = 30


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


class FraudPredictor:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = FraudClassifier(
            INPUT_SIZE
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=self.device
            )
        )

        self.model.eval()

        self.scaler = joblib.load(
            SCALER_PATH
        )

    def predict(self, transaction):
        """
        transaction must contain:

        Time,
        V1 through V28,
        Amount
        """

        transaction = np.array(
            transaction,
            dtype=np.float32
        )

        if len(transaction) != INPUT_SIZE:
            raise ValueError(
                f"Expected {INPUT_SIZE} features, "
                f"received {len(transaction)}."
            )

        # Time is index 0
        # Amount is index 29
        time_amount = np.array([
            [
                transaction[0],
                transaction[29]
            ]
        ])

        scaled = self.scaler.transform(
            time_amount
        )[0]

        transaction[0] = scaled[0]
        transaction[29] = scaled[1]

        tensor = torch.tensor(
            transaction,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor)
            score = torch.sigmoid(logit).item()

        is_fraud = score >= THRESHOLD

        return {
            "risk_score": score,
            "is_fraud": is_fraud,
            "threshold": THRESHOLD
        }