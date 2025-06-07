
"""
Model C : Multi-specialty Job Matching using Cosine Similarity
多专业岗位匹配度模型（余弦相似度）
"""

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df["text"])

majors = {
    "Computer Science": ["algorithm", "python", "data", "network"],
    "Education": ["teaching", "classroom", "student", "curriculum"],
    "Media": ["content", "video", "script", "publicity"]
}
results = []

for major_name, keywords in majors.items():
    major_vector = vectorizer.transform([" ".join(keywords)])
    similarity_scores = cosine_similarity(major_vector, X).flatten()
    df[f"Match_{major_name}"] = similarity_scores

df.to_csv("model_C_specialty_matching.csv", index=False)
print(" Model C completed: multi-specialty match scores saved to CSV.")
