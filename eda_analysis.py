
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("visualizations", exist_ok=True)

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 1. Dataset overview
print("\nDataset shape:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 2. Statistical summary
print("\nStatistical summary:\n")
print(df.describe().round(2))

# 3. Average scores by gender
print("\nAverage scores by gender:\n")
print(df.groupby("gender")[
    ["math score", "reading score", "writing score"]
].mean().round(2))

# 4. Average scores by test preparation
print("\nAverage scores by test preparation:\n")
print(df.groupby("test preparation course")[
    ["math score", "reading score", "writing score"]
].mean().round(2))

# 5. Score distributions
score_columns = ["math score", "reading score", "writing score"]

for column in score_columns:
    plt.figure(figsize=(7, 5))
    sns.histplot(df[column], bins=15, kde=True)
    plt.title(f"Distribution of {column.title()}")
    plt.xlabel("Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(
        f"visualizations/{column.replace(' ', '_')}_distribution.png",
        dpi=300
    )
    plt.close()

# 6. Correlation heatmap
plt.figure(figsize=(7, 5))
correlation = df[score_columns].corr()
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)
plt.title("Correlation Between Student Scores")
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png", dpi=300)
plt.close()

# 7. Scores by test preparation
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="test preparation course", y="writing score")
plt.title("Writing Scores by Test Preparation")
plt.xlabel("Test Preparation Course")
plt.ylabel("Writing Score")
plt.tight_layout()
plt.savefig("visualizations/writing_by_preparation.png", dpi=300)
plt.close()

# 8. Average scores by lunch type
lunch_means = df.groupby("lunch")[score_columns].mean()

lunch_means.plot(kind="bar", figsize=(8, 5))
plt.title("Average Scores by Lunch Type")
plt.xlabel("Lunch Type")
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("visualizations/scores_by_lunch.png", dpi=300)
plt.close()

print("\nEDA completed successfully.")
print("Graphs saved in the visualizations folder.")

# 9. Average scores by parental education
education_scores = df.groupby("parental level of education")[
    ["math score", "reading score", "writing score"]
].mean().sort_values("writing score")

print("\nAverage scores by parental education:\n")
print(education_scores.round(2))

education_scores.plot(kind="bar", figsize=(10, 6))
plt.title("Average Student Scores by Parental Education")
plt.xlabel("Parental Level of Education")
plt.ylabel("Average Score")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("visualizations/scores_by_parental_education.png", dpi=300)
plt.close()

# 10. Average scores by race/ethnicity
race_scores = df.groupby("race/ethnicity")[
    ["math score", "reading score", "writing score"]
].mean()

print("\nAverage scores by race/ethnicity:\n")
print(race_scores.round(2))

race_scores.plot(kind="bar", figsize=(9, 6))
plt.title("Average Student Scores by Race/Ethnicity")
plt.xlabel("Race/Ethnicity Group")
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("visualizations/scores_by_race_ethnicity.png", dpi=300)
plt.close()

print("\nAll EDA analyses and visualizations completed!")
