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

topic_dist_df = pd.DataFrame(
    lda_result,
    columns=[f"Topic_{i+1}" for i in range(lda.n_components)]
)
topic_dist_df["Position_ID"] = df.index
topic_dist_df["Industry"] = df["industry"]
topic_dist_df["LDA_Topic"] = df["LDA_Topic"]
topic_dist_df["Raw_Text"] = df["text"]


topic_dist_df.to_csv("output/Position_Topic_Distribution_Matrix.csv", index=False, encoding="utf-8-sig")
print(" 岗位主题分布矩阵 CSV 已保存！")

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(lda_result)

df.to_csv("model_A_lda_kmeans_output.csv", index=False)
print(" Model A completed: topic modeling and clustering saved to CSV.")
