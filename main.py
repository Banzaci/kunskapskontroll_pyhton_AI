import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Laddar in datasetet (CSV-fil) och returnerar det som en DataFrame
def load_housing():
  return pd.read_csv("data/housing.csv")

# Visar första 10 raderna i datasetet
def rows(df):
  print("\n--- Init ---")
  print(df.head(10))
  # Visar antal rader och kolumner
  print("\nShape:", df.shape)


def missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    for column in missing.index:
        median = df[column].median()
        df[column] = df[column].fillna(median)
        print(f"Filled {column} with median: {median}")

    print("\nMissing values before fill:\n", missing)

# Visar statistisk sammanfattning av datasetet
# (medelvärde, min, max, standardavvikelse osv.)
def stats(df):
  print("\n--- Describe ---")
  print(df.describe())

# hur många hus som finns i olika pris-intervall
def plot_price_distribution(df):
    plt.figure(figsize=(8,5))
    plt.title("House Price Distribution")
    plt.xlabel("Price")
    plt.show()


# Visar hur starkt varje variabel påverkar huspriset
# Högre värde = starkare påverkan
# Lägre/negativt värde = inte viktigt
def plot_correlation(df):
    corr = df.corr(numeric_only=True)["median_house_value"]
    # Sorterar så vi ser viktigast först
    corr = corr.sort_values(ascending=False)
    print("\nCorrelation with house price:\n")
    print(corr)


# Visar samband mellan inkomst och huspris
# (undersöker om rikare områden har dyrare hus)
def price_by_income(df):
    plt.figure(figsize=(8,5))
    plt.title("House Value vs Income")
    plt.xlabel("Median income")
    plt.ylabel("House value")
    plt.show()


# Delar upp datasetet i "billiga" och "dyra" hus
# antalet visas, tex cheap: 10323st
def cheap_vs_expensive(df):
    median = df["median_house_value"].median()

    df["price_category"] = np.where(
        df["median_house_value"] > median,
        "expensive",
        "cheap"
    )

    print("\n--- Number of cheap vs expensive  ---")
    print(df["price_category"].value_counts())

def remove_duplicates(df):
    before = len(df)

    df.drop_duplicates(inplace=True)

    after = len(df)

    print(f"\nRemoved {before - after} duplicate rows")

def avg_price_per_category(df):
    result = df.groupby("ocean_proximity")["median_house_value"].mean().sort_values(ascending=False)

    print("\n--- Average house price per ocean proximity category ---")
    print(result)

def ocean_distance_proxy(df):
    mapping = {
        "NEAR BAY": 1,
        "<1H OCEAN": 2,
        "NEAR OCEAN": 3,
        "ISLAND": 4,
        "INLAND": 5
    }

    df["ocean_distance_score"] = df["ocean_proximity"].map(mapping)

    result = df.groupby("ocean_distance_score")["median_house_value"].mean()

    print("\n--- Average price vs distance proxy (lower = closer to ocean) ---")
    print(result)

def conclusions(df):
    corr = df.corr(numeric_only=True)["median_house_value"]
    corr = corr.sort_values(ascending=False)

    print("\n--- Conclusions ---")
    print("Strongest positive factors:")
    print(corr.head())

    print("\nStrongest negative factors:")
    print(corr.tail())

def main():
    df = load_housing()
    missing_values(df)
    remove_duplicates(df)
    avg_price_per_category(df)
    ocean_distance_proxy(df)
    rows(df)
    stats(df)
    plot_price_distribution(df)
    plot_correlation(df)
    price_by_income(df)
    cheap_vs_expensive(df)
    conclusions(df)

if __name__ == "__main__":
  main()
