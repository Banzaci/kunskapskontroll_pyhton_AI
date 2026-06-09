import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

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
  # Visar hur många saknade värden som finns per kolumn
  missing = df.isnull().sum()
  missing = missing[missing > 0]

  print("\nMissing values:\n", missing)

# Visar statistisk sammanfattning av datasetet
# (medelvärde, min, max, standardavvikelse osv.)
def stats(df):
  print("\n--- Describe ---")
  print(df.describe())

# hur många hus som finns i olika pris-intervall
def plot_price_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df["median_house_value"], bins=50, kde=True)
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
    sns.scatterplot(
        x=df["median_income"],
        y=df["median_house_value"],
        alpha=0.3
    )
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


def main():
    df = load_housing()

    # rows(df)
    missing_values(df)
    # stats(df)

    # plot_price_distribution(df)
    # plot_correlation(df)
    # price_by_income(df)

    cheap_vs_expensive(df)

if __name__ == "__main__":
  main()
