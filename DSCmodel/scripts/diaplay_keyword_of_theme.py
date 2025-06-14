import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

custom_stopwords = list(set(ENGLISH_STOP_WORDS).union({
    "000", "good", "make", "job", "salary","team", "customer", "work","hours",
    "requirements","time","able","working","hours","long","unlimited","method","company"
}))

data_dir = "D:/Users/WU/Desktop/DSCmodel/data"
output_dir = "D:/Users/WU/Desktop/DSCmodel/output"
n_topics = 5
n_top_words = 10

files = {
    "education": "Educational_position.json",
    "culture": "Culture_medium.json",
    "electronic": "Electronic_information.json",
    "mechanical": "Mechanical_Finance.json"
}

for industry, filename in files.items():
    file_path = os.path.join(data_dir, filename)
    print(f"\n正在处理：{industry} - {filename}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["text"] = df["Word segmentation_string"].fillna("")

    vectorizer = CountVectorizer(max_df=0.85, min_df=3, stop_words=custom_stopwords)

    X = vectorizer.fit_transform(df["text"])
    feature_names = vectorizer.get_feature_names_out()

    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(X)

    topics_result = []
    for topic_idx, topic_weights in enumerate(lda_model.components_):
        top_indices = topic_weights.argsort()[::-1][:n_top_words]
        top_words = [(feature_names[i], topic_weights[i]) for i in top_indices]
        topics_result.append([word for word, _ in top_words])

        top_words_str = ", ".join([f"{word} ({weight:.2f})" for word, weight in top_words])
        print(f" Topic {topic_idx + 1}: {top_words_str}")

    df_topics = pd.DataFrame(
        topics_result,
        index=[f"Topic {i+1}" for i in range(n_topics)],
        columns=[f"Top {i+1}" for i in range(n_top_words)]
    )
    output_path = os.path.join(output_dir, f"lda_keywords_{industry}.csv")
    df_topics.to_csv(output_path, index=True, encoding="utf-8-sig")
    print(f"已保存至：{output_path}")
