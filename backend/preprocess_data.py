import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CSV_PATH = "data/creditcard.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    # -----------------------------
    # Basic dataset inspection
    # -----------------------------
    print("Dataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentages:")
    print(df["Class"].value_counts(normalize=True) * 100)

    # -----------------------------
    # Separate features and target
    # -----------------------------
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # -----------------------------
    # Train / validation / test
    # 70% train
    # 15% validation
    # 15% test
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

    print("\nDataset splits:")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    print("\nFraud cases per split:")
    print(f"Train:      {y_train.sum()}")
    print(f"Validation: {y_val.sum()}")
    print(f"Test:       {y_test.sum()}")

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

    print("\nScaled training data preview:")
    print(X_train.head())


if __name__ == "__main__":
    main()