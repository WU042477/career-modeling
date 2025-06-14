import json
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS
custom_stopwords = STOPWORDS.union({
    "job", "work", "live", "good", "requirement", "able", "etc", "provide", "company"
})

json_path = r"D:\Users\WU\Desktop\DSCmodel\data\Educational_position.json"
font_path = r"C:\Windows\Fonts\msyh.ttc"  

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

output_path = os.path.join(os.path.dirname(json_path), "wordcloud_output_education.png")
wordcloud.to_file(output_path)
print(f"词云图已保存至：{output_path}")
