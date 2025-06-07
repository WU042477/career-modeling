
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
# Step 1: Load and prepare job data / 加载岗位数据
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

# Step 2: Topic modeling using LDA / 使用LDA进行主题建模
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
X = vectorizer.fit_transform(df["text"])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_result = lda.fit_transform(X)
df["LDA_Topic"] = lda_result.argmax(axis=1)

# Step 3: Clustering based on topic distribution / 基于主题分布进行KMeans聚类
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(lda_result)

# Step 4: Output clustered topics / 输出聚类结果
df.to_csv("model_A_lda_kmeans_output.csv", index=False)
print(" Model A completed: topic modeling and clustering saved to CSV.")
