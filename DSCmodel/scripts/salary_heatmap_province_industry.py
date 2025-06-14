import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import os

file_paths = {
    "Education": "data/Educational_position.json",
    "Culture": "data/Culture_medium.json",
    "Electronic": "data/Electronic_information.json",
    "Mechanical_Finance": "data/Mechanical_Finance.json"
}

all_data = []
for industry, path in file_paths.items():
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
        for rec in records:
            rec["industry"] = industry
        all_data.extend(records)

df = pd.DataFrame(all_data)

df["text"] = df["Word segmentation_string"].fillna("")
df["salary_avg"] = pd.to_numeric(df["salary_avg"], errors="coerce")
df["province"] = df["Province/Municipality/Autonomous Region (County)"].fillna("Unknown")

vectorizer = TfidfVectorizer(max_features=100)
X_text = vectorizer.fit_transform(df["text"])

grouped = df.groupby(["province", "industry"])["salary_avg"].mean().reset_index()
pivot_table = grouped.pivot(index="province", columns="industry", values="salary_avg")

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5)
plt.title("Average Salary by Province and Industry", fontsize=16)
plt.ylabel("Province")
plt.xlabel("Industry")
plt.tight_layout()
plt.savefig("output/salary_heatmap_province_industry.png")
plt.close()
