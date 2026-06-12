import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------- LOAD ----------------
# Läser in housing-datasetet från CSV-fil
def load_housing():
    return pd.read_csv("data/housing.csv")


# ---------------- CLEANING ----------------

# Hanterar saknade värden i datasetet
# Fyller numeriska kolumner med medianvärdet
def missing_values(df):
    missing = df.isnull().sum()          # räknar NaN per kolumn
    missing = missing[missing > 0]       # filtrerar endast kolumner med saknade värden

    print("\nMissing BEFORE fill:\n", missing)

    # Loopar igenom alla kolumner med saknade värden
    for column in missing.index:
        if pd.api.types.is_numeric_dtype(df[column]):  # endast numeriska kolumner
            median = df[column].median()                # beräknar median
            df[column] = df[column].fillna(median)      # fyller saknade värden
            print(f"Filled {column} with median: {median}")


# Tar bort eventuella duplicerade rader i datasetet
def remove_duplicates(df):
    before = len(df)                # antal rader före
    df.drop_duplicates(inplace=True)
    after = len(df)                 # antal rader efter

    print(f"\nRemoved {before - after} duplicate rows")


# Kodar om kategorisk variabel till numeriska värden
# (ocean proximity -> integer labels)
def encode_categorical(df):
    df["ocean_proximity"] = df["ocean_proximity"].astype("category").cat.codes


# ---------------- EXPLORATORY DATA ANALYSIS (EDA) ----------------

# Visar första raderna och datats struktur
def rows(df):
    print("\n--- INIT ---")
    print(df.head(10))      # första 10 rader
    print("\nShape:", df.shape)  # antal rader och kolumner


# Visar statistisk sammanfattning av datasetet
# (mean, min, max, std osv.)
def stats(df):
    print("\n--- DESCRIBE ---")
    print(df.describe())


# Visar fördelning av huspriser
def plot_price_distribution(df):
    plt.figure(figsize=(8,5))
    plt.hist(df["median_house_value"], bins=50)
    plt.title("House Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.show()


# Visar korrelation mellan alla variabler och huspris
def plot_correlation(df):
    corr = df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)

    print("\n--- Correlation with house price ---")
    print(corr)


# Visar samband mellan inkomst och huspris
def price_by_income(df):
    plt.figure(figsize=(8,5))
    plt.scatter(df["median_income"], df["median_house_value"], alpha=0.3)
    plt.title("House Value vs Income")
    plt.xlabel("Median income")
    plt.ylabel("House value")
    plt.show()


# Delar upp hus i dyra och billiga baserat på medianpris
def cheap_vs_expensive(df):
    median = df["median_house_value"].median()

    df["price_category"] = np.where(
        df["median_house_value"] > median,
        "expensive",
        "cheap"
    )

    print("\n--- Cheap vs Expensive ---")
    print(df["price_category"].value_counts())


# ---------------- ANALYSIS ----------------

# Räknar genomsnittligt huspris per havsnärhetskategori
def avg_price_per_category(df):
    df["ocean_proximity_raw"] = df["ocean_proximity"]  # sparar original innan encoding

    result = df.groupby("ocean_proximity_raw")["median_house_value"].mean().sort_values(ascending=False)

    print("\n--- Average price per ocean category ---")
    print(result)


# Skapar en "proxy" för avstånd till havet
# (lägre värde = närmare havet)
def ocean_distance_proxy(df):
    mapping = {
        "NEAR BAY": 1,
        "<1H OCEAN": 2,
        "NEAR OCEAN": 3,
        "ISLAND": 4,
        "INLAND": 5
    }

    df["ocean_distance_score"] = df["ocean_proximity_raw"].map(mapping)

    result = df.groupby("ocean_distance_score")["median_house_value"].mean()

    print("\n--- Price vs distance proxy ---")
    print(result)


# Sammanfattar vilka faktorer som påverkar huspris mest
def conclusions(df):
    corr = df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)

    print("\n--- CONCLUSIONS ---")
    print("Top positive factors:")
    print(corr.head())

    print("\nTop negative factors:")
    print(corr.tail())


# ---------------- MAIN ----------------
def main():
    df = load_housing()

    missing_values(df)        # steg 1: saknade värden
    remove_duplicates(df)     # steg 2: dubletter

    encode_categorical(df)    # steg 3: encoding

    rows(df)                  # snabb översikt
    stats(df)                 # statistik

    plot_price_distribution(df)  # fördelning
    price_by_income(df)          # inkomst vs pris
    plot_correlation(df)         # korrelationer

    avg_price_per_category(df)   # havsnärhet vs pris
    ocean_distance_proxy(df)     # avståndsanalys

    cheap_vs_expensive(df)       # klassificering
    conclusions(df)              # slutsats

if __name__ == "__main__":
    main()