
"""
=========================================================
WEEK 1 - BEAUTY & WELLNESS DATA SCIENCE PROJECT
Project: Skincare Customer Preference & Product Review
Analysis

Objective:
Analyze skincare customer preferences, product ratings,
skin types, skin concerns and product recommendations.

Note:
1. Dataset contains synthetic/demo records.
2. Use properly licensed public data for real conclusions.
3. This project is for business analytics demonstration,
   not medical or dermatological advice.
=========================================================
"""

# 1. IMPORT LIBRARIES

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# 2. PROJECT PATH CONFIGURATION

ROOT = Path(__file__).resolve().parent

DATA_PATH = Path(
    r"C:\Users\Adarsh\Downloads\Week1_Skincare_DataScience_Project (1)\Week1_Skincare_DataScience_Project\data\sample_skincare_reviews.csv"
)

OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 3. LOAD DATASET

def load_data(path=DATA_PATH):

    print("\nLoading dataset...")

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "product_name",
        "skin_type",
        "skin_concern",
        "price_inr",
        "rating",
        "is_recommended",
        "ingredients"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print("Dataset loaded successfully!")

    return df


# 4. DATA CLEANING

def clean_data(df):

    print("\nCleaning dataset...")

    df = df.copy()

    # Convert columns into numeric format

    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    )

    df["price_inr"] = pd.to_numeric(
        df["price_inr"], errors="coerce"
    )

    df["is_recommended"] = pd.to_numeric(
        df["is_recommended"], errors="coerce"
    )

    # Remove incomplete records

    df = df.dropna(
        subset=["product_name", "skin_type", "rating"]
    )

    # Keep valid rating values between 1 and 5

    df = df[df["rating"].between(1, 5)]

    if df.empty:
        raise ValueError(
            "No valid records available after cleaning."
        )

    print("Data cleaning completed!")

    return df


# 5. EXPLORATORY DATA ANALYSIS

def summarize(df):

    print("\n" + "=" * 55)
    print("             DATASET OVERVIEW")
    print("=" * 55)

    print(f"\nTotal Records: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")

    # Missing values

    print("\nMissing Values:")

    print(df.isnull().sum())

    # Average rating by skin type

    print("\nAverage Rating by Skin Type:")

    rating_summary = (
        df.groupby("skin_type")["rating"]
        .agg(["count", "mean"])
        .round(2)
    )

    print(rating_summary)

    # Recommendation rate

    print("\nRecommendation Rate by Skin Type (%):")

    recommendation_rate = (
        df.groupby("skin_type")["is_recommended"]
        .mean()
        .mul(100)
        .round(2)
    )

    print(recommendation_rate)

    # Top rated products

    print("\nTop 10 Products by Average Rating:")

    product_summary = (
        df.groupby("product_name")
        .agg(
            review_count=("rating", "size"),
            avg_rating=("rating", "mean")
        )
        .query("review_count >= 2")
        .sort_values(
            ["avg_rating", "review_count"],
            ascending=False
        )
    )

    print(product_summary.head(10).round(2))

    # Average product price

    print("\nAverage Product Price:")

    print(
        f"INR {df['price_inr'].mean():.2f}"
    )

    return product_summary


# 6. DATA VISUALIZATION

def create_charts(df):

    print("\nCreating visualizations...")

    # Chart 1: Average rating by skin type

    rating_data = (
        df.groupby("skin_type")["rating"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(8, 5))

    rating_data.plot(kind="bar")

    plt.title("Average Product Rating by Skin Type")
    plt.xlabel("Skin Type")
    plt.ylabel("Average Rating (1-5)")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "average_rating_by_skin_type.png",
        dpi=160
    )

    plt.close()

    # Chart 2: Records by skin concern

    concern_data = (
        df.groupby("skin_concern")
        .size()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))

    concern_data.plot(kind="bar")

    plt.title("Number of Records by Skin Concern")
    plt.xlabel("Skin Concern")
    plt.ylabel("Number of Records")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "records_by_skin_concern.png",
        dpi=160
    )

    plt.close()

    # Chart 3: Product price distribution

    plt.figure(figsize=(8, 5))

    df["price_inr"].dropna().plot(
        kind="hist",
        bins=10
    )

    plt.title("Skincare Product Price Distribution")
    plt.xlabel("Price (INR)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "product_price_distribution.png",
        dpi=160
    )

    plt.close()

    print("All charts saved successfully!")


# 7. SAVE CLEANED DATASET

def save_cleaned_data(df):

    output_file = OUTPUT_DIR / "cleaned_skincare_data.csv"

    df.to_csv(output_file, index=False)

    print(f"\nCleaned dataset saved: {output_file}")


# 8. MAIN FUNCTION

def main():

    print("\n" + "=" * 55)
    print("       SKINCARE DATA SCIENCE PROJECT")
    print("                    WEEK 1")
    print("=" * 55)

    try:

        # Load dataset

        df = load_data()

        # Clean dataset

        df = clean_data(df)

        # Perform analysis

        summarize(df)

        # Generate charts

        create_charts(df)

        # Save cleaned dataset

        save_cleaned_data(df)

        print("\n" + "=" * 55)
        print("       PROJECT COMPLETED SUCCESSFULLY")
        print("=" * 55)

        print(f"\nOutput Folder: {OUTPUT_DIR}")

        print(
            "\nNote: Dataset contains synthetic/demo records. "
            "Interpret findings cautiously."
        )

    except Exception as error:

        print("\nPROJECT ERROR:")
        print(error)


# 9. PROGRAM EXECUTION

if __name__ == "__main__":
    main()
