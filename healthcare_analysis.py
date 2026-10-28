
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("results", exist_ok=True)

# Find dataset file
# Find dataset files in current and nested folders
files = glob.glob("**/*.tsv", recursive=True)
files += glob.glob("**/*.csv", recursive=True)

files = [
    f for f in files
    if "train" in os.path.basename(f).lower()
]
if not files:
    raise FileNotFoundError(
        "Place your drug review CSV or TSV file "
        "inside this project folder."
    )

# Load dataset
df = pd.read_csv(files[0], sep=None, engine="python")
df.columns = df.columns.str.strip()

print("Dataset:", files[0])
print("\nAvailable columns:")
print(df.columns.tolist())

# Identify columns
columns = {c.lower(): c for c in df.columns}

rating_col = columns.get("rating")
if not rating_col:
    raise ValueError("No rating column found in the dataset.")

condition_col = columns.get("condition")

# Find available patient review columns
review_cols = [
    c for c in df.columns
    if any(word in c.lower() for word in
           ["review", "comment", "benefit", "sideeffect"])
]

if not review_cols:
    raise ValueError("No review text column found.")

# Prepare data
df[rating_col] = pd.to_numeric(
    df[rating_col], errors="coerce"
)
df = df.dropna(subset=[rating_col]).copy()

df["Review_Text"] = (
    df[review_cols].fillna("").astype(str)
    .agg(" ".join, axis=1).str.strip()
)

df = df[df["Review_Text"] != ""].copy()

# Classify sentiment from ratings
df["Sentiment"] = df[rating_col].apply(
    lambda rating: "Positive" if rating >= 7
    else "Neutral" if rating >= 4
    else "Negative"
)

# Save analyzed dataset
df.to_csv("results/analyzed_reviews.csv", index=False)

# Display results
print("\nFirst 5 records:")
print(df[["Review_Text", rating_col, "Sentiment"]].head())

print("\nSentiment counts:")
print(df["Sentiment"].value_counts())

print("\nAverage rating:", round(df[rating_col].mean(), 2))
print("Total reviews:", len(df))

# Chart 1: Sentiment distribution
sns.countplot(
    data=df, x="Sentiment",
    order=["Positive", "Neutral", "Negative"]
)
plt.title("Patient Drug Review Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("results/sentiment_chart.png")
plt.show()
plt.close()

# Chart 2: Rating distribution
sns.histplot(df[rating_col], bins=10, kde=False)
plt.title("Distribution of Patient Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("results/rating_chart.png")
plt.show()
plt.close()

# Chart 3: Most reviewed health conditions
if condition_col:
    top_conditions = df[condition_col].dropna().value_counts().head(10)

    if not top_conditions.empty:
        top_conditions.sort_values().plot(kind="barh")
        plt.title("Top 10 Reviewed Health Conditions")
        plt.xlabel("Number of Reviews")
        plt.ylabel("Health Condition")
        plt.tight_layout()
        plt.savefig("results/condition_chart.png")
        plt.show()
        plt.close()

# Save a summary for the report
with open("results/summary.txt", "w", encoding="utf-8") as f:
    f.write("PATIENT DRUG REVIEW ANALYSIS\n")
    f.write(f"Total reviews analyzed: {len(df)}\n")
    f.write(f"Average rating: {df[rating_col].mean():.2f}\n\n")
    f.write("Sentiment distribution:\n")
    f.write(df["Sentiment"].value_counts().to_string())

print("\nSUCCESS! Check the results folder for charts,")
print("analyzed_reviews.csv and summary.txt.")
