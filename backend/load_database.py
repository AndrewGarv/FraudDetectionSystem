import sqlite3
import pandas as pd


CSV_PATH = "data/creditcard.csv"
DB_PATH = "database/fraud_detection.db"


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} transactions.")

    connection = sqlite3.connect(DB_PATH)

    print("Writing transactions to SQLite...")

    df.to_sql(
        "transactions",
        connection,
        if_exists="replace",
        index=False
    )

    connection.close()

    print("Database created successfully.")
    print(f"Location: {DB_PATH}")


if __name__ == "__main__":
    main()