
"""
Model C (Advanced): Multi-specialty Job Matching using Cosine Similarity
模型C（升级）：多专业岗位匹配度模型（余弦相似度）
"""

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load job data / 加载岗位数据
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

# Step 2: TF-IDF encoding / 岗位TF-IDF编码
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df["text"])

# Step 3: Define multiple specialties / 定义多个专业
majors = {
    "Computer Science": ["algorithm", "python", "data", "network"],
    "Education": ["teaching", "classroom", "student", "curriculum"],
    "Media": ["content", "video", "script", "publicity"]
}
results = []

# Step 4: Calculate cosine similarity / 计算与岗位的匹配分数
for major_name, keywords in majors.items():
    major_vector = vectorizer.transform([" ".join(keywords)])
    similarity_scores = cosine_similarity(major_vector, X).flatten()
    df[f"Match_{major_name}"] = similarity_scores

# Step 5: Output match matrix / 输出匹配矩阵
df.to_csv("model_C_specialty_matching.csv", index=False)
print("✅ Model C completed: multi-specialty match scores saved to CSV.")
