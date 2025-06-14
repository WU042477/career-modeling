import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def visualize_salary_prediction(filepath):
    df = pd.read_json(filepath, encoding="utf-8")
    df = df[df["salary_avg"].notna()]

    if "Word segmentation_string" not in df.columns or "salary_avg" not in df.columns:
        print(f" Missing required fields in {filepath}")
        return None

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["Word segmentation_string"])

    model = LinearRegression()
    model.fit(X, df["salary_avg"])
    y_pred = model.predict(X)
    y_true = df["salary_avg"]

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--')
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title(f"{os.path.basename(filepath)}\nActual vs Predicted Salary")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename}_salary_plot.png", dpi=300)
    plt.show()

    print(f" {os.path.basename(filepath)}")
    print(f" MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f" R² Score: {r2_score(y_true, y_pred):.3f}")
    print("-" * 50)

base_path = r"D:\Users\WU\Desktop\DSCmodel\data"

files = [
    "Culture_medium.json",
    "Educational_position.json",
    "Electronic_information.json",
    "Mechanical_Finance.json"
]

for filename in files:
    filepath = os.path.join(base_path, filename)
    visualize_salary_prediction(filepath)
