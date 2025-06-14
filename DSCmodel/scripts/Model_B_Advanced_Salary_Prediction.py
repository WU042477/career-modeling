"""
Model B : Salary Prediction using Regression
岗位薪资回归预测模型（含图表保存）
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("output", exist_ok=True)

files = {
    "Education": "data/Educational_position.json",
    "Culture": "data/Culture_medium.json",
    "Electronic": "data/Electronic_information.json",
    "Mechanical_Finance": "data/Mechanical_Finance.json"
}

records = []
for industry, file in files.items():
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            item["industry"] = industry
        records.extend(data)

df = pd.DataFrame(records)
df["text"] = df["Word segmentation_string"].fillna("")
df["salary_avg"] = pd.to_numeric(df["salary_avg"], errors="coerce")

vectorizer = TfidfVectorizer(max_features=100)
X_text = vectorizer.fit_transform(df["text"])
X_df = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())
X_df["industry_code"] = df["industry"].astype("category").cat.codes
X = X_df
y = df["salary_avg"].fillna(df["salary_avg"].mean())

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
df["Predicted_Salary"] = y_pred

output_csv_path = "output/model_B_salary_prediction.csv"
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
print(f" 薪资预测结果已保存：{output_csv_path}")

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="industry", y="Predicted_Salary", estimator='mean', ci=None, palette="Set2")
plt.title("Average Predicted Salary by Industry", fontsize=14)
plt.ylabel("Predicted Salary (RMB)", fontsize=12)
plt.xlabel("Industry", fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
industry_plot_path = "output/avg_predicted_salary_by_industry.png"
plt.savefig(industry_plot_path, dpi=300)
plt.show()
print(f" 图表已保存：{industry_plot_path}")

coefficients = model.coef_[:100]  
feature_names = vectorizer.get_feature_names_out()
top_n = 20
top_indices = np.argsort(np.abs(coefficients))[-top_n:]
top_features = feature_names[top_indices]
top_weights = coefficients[top_indices]

plt.figure(figsize=(10, 6))
plt.barh(top_features, top_weights, color=['#2ca02c' if w > 0 else '#d62728' for w in top_weights])
plt.xlabel("Coefficient Weight")
plt.title("Top 20 Keyword Coefficients Affecting Salary")
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
keyword_plot_path = "output/top_20_keyword_coefficients.png"
plt.savefig(keyword_plot_path, dpi=300)
plt.show()
print(f"关键词系数图已保存：{keyword_plot_path}")
