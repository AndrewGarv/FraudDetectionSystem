import pandas as pd


DATA_PATH = "data/creditcard.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 transactions:")
    print(df.head())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentages:")
    print(df["Class"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()