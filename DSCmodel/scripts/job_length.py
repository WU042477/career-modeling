import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

json_path = r"D:\Users\WU\Desktop\DSCmodel\data\Culture_medium.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)

df["text_length"] = df["Word segmentation_string"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

plt.figure(figsize=(10, 6))
plt.hist(df["text_length"], bins=30, color='skyblue', edgecolor='black')
plt.title("The number of vocabulary items and the number of positions", fontsize=16)
plt.xlabel("The number of vocabulary", fontsize=12)
plt.ylabel("The number of position", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("D:/Users/WU/Desktop/DSCmodel/output/job_length_hist.png")
plt.show()

