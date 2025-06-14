import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "Microsoft YaHei"

json_path = r"D:\Users\WU\Desktop\DSCmodel\data\Educational_position.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)

df["text_length"] = df["Word segmentation_string"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
df = df[df["salary_avg"].notna() & df["text_length"].notna()]

plt.figure(figsize=(10, 6))
plt.scatter(df["text_length"], df["salary_avg"], alpha=0.6, color='purple', edgecolor='k')
plt.title("Education | Word Count vs. Average Salary", fontsize=16)
plt.xlabel("Word Count", fontsize=12)
plt.ylabel("Average Salary (CNY/month)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

output_path = r"D:\Users\WU\Desktop\DSCmodel\output\job_length_vs_salary_education.png"
plt.savefig(output_path)
print(f"Plot saved to: {output_path}")
plt.show()
