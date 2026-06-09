import pandas as pd

from utils.wordcloud_builder import build_wordcloud

def load_data():

    df = pd.read_csv(
        "analysis_data.csv",
        encoding="utf-8-sig"
    )

    df["comment_star"] = pd.to_numeric(
        df["comment_star"],
        errors="coerce"
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    return df


def build_all_wordclouds(df):
    products = df["product_name"].unique()
    for product in products:
        print(f"生成词云：{product}")
        sub = df[df["product_name"] == product]
        # 传入 word_segment 列
        build_wordcloud(sub["word_segment"], product)
    print("全部词云生成完成")