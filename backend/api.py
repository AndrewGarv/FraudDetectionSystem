import sqlite3
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

from predict import FraudPredictor

DB_PATH = "database/fraud_detection.db"

app = FastAPI(
    title="Fraud Detection API",
    description="PyTorch-powered credit card fraud detection API",
    version="1.0.0"
)
# Space
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


predictor = FraudPredictor()

# -------------------------
# Database helper functions
# -------------------------

def initialize_prediction_table():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            amount REAL NOT NULL,
            risk_score REAL NOT NULL,
            is_fraud INTEGER NOT NULL,
            threshold REAL NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def log_prediction(amount, risk_score, is_fraud, threshold):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO prediction_logs (
            created_at,
            amount,
            risk_score,
            is_fraud,
            threshold
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        amount,
        risk_score,
        int(is_fraud),
        threshold
    ))

    connection.commit()
    connection.close()


initialize_prediction_table()


# -------------------------
# Request model
# -------------------------


class TransactionRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):

    features = [
        transaction.Time,
        transaction.V1,
        transaction.V2,
        transaction.V3,
        transaction.V4,
        transaction.V5,
        transaction.V6,
        transaction.V7,
        transaction.V8,
        transaction.V9,
        transaction.V10,
        transaction.V11,
        transaction.V12,
        transaction.V13,
        transaction.V14,
        transaction.V15,
        transaction.V16,
        transaction.V17,
        transaction.V18,
        transaction.V19,
        transaction.V20,
        transaction.V21,
        transaction.V22,
        transaction.V23,
        transaction.V24,
        transaction.V25,
        transaction.V26,
        transaction.V27,
        transaction.V28,
        transaction.Amount
    ]

    result = predictor.predict(features)

    log_prediction(
        amount=transaction.Amount,
        risk_score=result["risk_score"],
        is_fraud=result["is_fraud"],
        threshold=result["threshold"]
    )

    return result

@app.get("/predictions")
def get_predictions():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            created_at,
            amount,
            risk_score,
            is_fraud,
            threshold
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT 100
    """)

    rows = cursor.fetchall()
    connection.close()

    predictions = []

    for row in rows:
        predictions.append({
            "id": row[0],
            "created_at": row[1],
            "amount": row[2],
            "risk_score": row[3],
            "is_fraud": bool(row[4]),
            "threshold": row[5]
        })

    return predictions   


@app.get("/stats")
def get_stats():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            COUNT(*),
            SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END),
            AVG(risk_score)
        FROM prediction_logs
    """)

    row = cursor.fetchone()
    connection.close()

    return {
        "total_predictions": row[0] or 0,
        "fraud_detected": row[1] or 0,
        "legitimate": row[2] or 0,
        "average_risk_score": row[3] or 0
    }