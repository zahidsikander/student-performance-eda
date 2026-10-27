
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Create output folder if it does not exist
os.makedirs("visualization", exist_ok=True)

# 2. Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 3. Select features and target
X = df.drop(columns=["writing score"])
y = df["writing score"]

# 4. Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["str", "object"]).columns
numerical_cols = X.select_dtypes(exclude=["str", "object"]).columns

# 5. Prepare data preprocessing
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("numerical", "passthrough", numerical_cols)
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Define machine learning models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2
    )
}
results = []
linear_predictions = None

# 8. Train, predict, and evaluate each model
for name, algorithm in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", algorithm)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])

    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

    if name == "Linear Regression":
        linear_predictions = predictions

# 9. Create and save model comparison graph
results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "RMSE", "R2"]
)

results_df.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar", figsize=(8, 5)
)

plt.title("Model Performance Comparison")
plt.ylabel("Error (lower is better)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("visualization/model_comparison.png", dpi=300)
plt.show()
plt.close()

# 10. Create and save actual vs predicted graph
plt.figure(figsize=(8, 5))
plt.scatter(y_test, linear_predictions, alpha=0.7)

# Reference line: perfect predictions
plt.plot([0, 100], [0, 100], linestyle="--")

plt.xlabel("Actual Writing Scores")
plt.ylabel("Predicted Writing Scores")
plt.title("Linear Regression: Actual vs Predicted")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.savefig("visualization/actual_vs_predicted.png", dpi=300)
plt.show()
plt.close()

# 11. Confirm saved graphs
print("\nGraphs saved successfully:")
print("visualization/model_comparison.png")
print("visualization/actual_vs_predicted.png")
