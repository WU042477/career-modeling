
"""
Model B : Salary Prediction using Regression
岗位薪资回归预测模型
"""

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

files = {
    "Education": "Educational_position.json",
    "Culture": "Culture_medium.json",
    "Electronic": "Electronic_information.json",
    "Mechanical_Finance": "Mechanical_Finance.json"
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
X = pd.concat([X_df], axis=1)
y = df["salary_avg"].fillna(df["salary_avg"].mean())

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

df["predicted_salary"] = y_pred
df.to_csv("model_B_salary_prediction.csv", index=False)
print("Model B completed: regression prediction saved to CSV.")
