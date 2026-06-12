import pandas as pd

from utils.wordcloud_builder import build_wordcloud
from utils.cluster_builder import get_cluster_data
from utils.cache import (
    get_product_summary_cache, get_cluster_cache
)

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




# 预热所有数据（聚类+AI总结）
def warmup_all(df):
    products = df["product_name"].unique().tolist()

    print(f"\n{'='*50}")
    print(f"[预热] 共 {len(products)} 个商品")
    print(f"{'='*50}")
    
    # 词云预热（直接调用生成函数，内部会自动跳过已有的缓存）
    build_all_wordclouds(df)

    # 聚类预热
    warmup_cluster_data(df)



# 生成词云：遍历所有商品，批量生成词云并写入缓存
def build_all_wordclouds(df):
    products = df["product_name"].unique()
    for product in products:
        print(f"生成词云：{product}")
        sub = df[df["product_name"] == product]
        # 传入 word_segment 列
        build_wordcloud(sub["word_segment"], product)
    print("全部词云生成完成")



# ─────────────────────────────────────────────
#  预热聚类：启动时批量生成所有产品的聚类+AI总结
# ─────────────────────────────────────────────
def warmup_cluster_data(df):
    """
    遍历所有产品，提前跑完聚类和AI总结并写入缓存。
    在 app.py 启动时调用一次即可，后续请求全部命中缓存。
 
    Args:
        df: 全局 DataFrame
    """
    products = df["product_name"].unique().tolist()
    total = len(products)
    print(f"[预热] 开始聚类预热，共 {total} 个产品...")
 
    for idx, product in enumerate(products, 1):
        # 已有缓存则跳过，支持断点续跑
        if get_cluster_cache(product):
            print(f"[预热] ({idx}/{total}) {product} 已有缓存，跳过")
            continue
 
        print(f"[预热] ({idx}/{total}) 正在处理：{product}")
        sub = df[df["product_name"] == product]
        try:
            get_cluster_data(sub)  # 内部自动写缓存
            print(f"[预热] ({idx}/{total}) {product} 完成")
        except Exception as e:
            print(f"[预热] ({idx}/{total}) {product} 失败：{e}")
 
    print(f"[预热] 聚类预热完成")
