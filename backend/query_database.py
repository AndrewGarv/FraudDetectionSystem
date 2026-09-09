import sqlite3


DB_PATH = "database/fraud_detection.db"


def main():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM transactions
    """)

    total_transactions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM transactions
        WHERE Class = 1
    """)

    fraud_transactions = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM transactions
        WHERE Class = 0
    """)

    legitimate_transactions = cursor.fetchone()[0]

    print(f"Total transactions: {total_transactions}")
    print(f"Fraudulent transactions: {fraud_transactions}")
    print(f"Legitimate transactions: {legitimate_transactions}")

    connection.close()


if __name__ == "__main__":
    main()