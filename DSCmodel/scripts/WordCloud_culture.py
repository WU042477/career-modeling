import json
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS
custom_stopwords = STOPWORDS.union({
    "job", "work", "live", "good", "requirement", "able", "etc", "provide", "company"
})

json_path = r"D:\Users\WU\Desktop\DSCmodel\data\Culture_medium.json"
font_path = r"C:\Windows\Fonts\msyh.ttc"  
output_dir = r"D:\Users\WU\Desktop\DSCmodel\output"
os.makedirs(output_dir, exist_ok=True)  
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

all_tokens = [token for tokens in df["clean_tokens"] for token in tokens if isinstance(tokens, list)]

text = " ".join(all_tokens)

wordcloud = WordCloud(
    font_path=font_path,
    width=1200,
    height=800,
    background_color="white",
    max_words=200,
    stopwords=custom_stopwords, 
    collocations=False
).generate(text)

plt.figure(figsize=(14, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Cultural Media Keyword WordCloud", fontsize=20)
plt.tight_layout()
plt.show()

output_path = os.path.join(output_dir, "wordcloud_output_culture.png")
wordcloud.to_file(output_path)
print(f" 词云图已成功保存到：{output_path}")