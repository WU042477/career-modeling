
"""
Model A : Keyword Topic Modeling + Clustering
关键词主题建模 + 聚类分析
"""
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
X = vectorizer.fit_transform(df["text"])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_result = lda.fit_transform(X)
df["LDA_Topic"] = lda_result.argmax(axis=1)

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(lda_result)

df.to_csv("model_A_lda_kmeans_output.csv", index=False)
print(" Model A completed: topic modeling and clustering saved to CSV.")
