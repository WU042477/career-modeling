import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Users\WU\Desktop\DSCmodel\output\Position_Topic_Distribution_Matrix.csv")

topic_cols = [col for col in df.columns if col.startswith("Topic_")]
X = df[topic_cols]

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)
df["TSNE_1"] = X_2d[:, 0]
df["TSNE_2"] = X_2d[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="TSNE_1", y="TSNE_2", hue="Cluster", palette="Set2", s=60)
plt.title("Position theme distribution clustering visualization", fontsize=14)
plt.legend(title="Cluster")
plt.show()

cluster_theme_avg = df.groupby("Cluster")[topic_cols].mean()

plt.figure(figsize=(8, 5))
sns.heatmap(cluster_theme_avg, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("The average proportion of each type of position on different topics", fontsize=14)
plt.xlabel("Topic")
plt.ylabel("Cluster")
plt.show()
df.to_csv("output/job_theme.csv", index=False, encoding="utf-8-sig")
