"""
product_service.py —— 产品数据查询层

职责边界（只做这三件事）：
  1. 持有全局 DataFrame（通过 init_service 注入）
  2. 对外提供产品列表、基础信息、结构化摘要的查询接口
  3. 启动时预热所有产品数据（调 data_processor / cache，自己不写缓存逻辑）

不做的事：
  - 不自己拼 prompt（ai_client 封装调用，prompt 在本文件内集中管理）
  - 不自己管缓存文件（cache.py 的事）
  - 不做聚类计算（data_processor.py 的事）
"""

from .cache import (
    get_product_summary_cache, set_product_summary_cache, get_cluster_cache, set_cluster_cache
)
from .data_processor import get_radar_data

_df = None   # 全局 DataFrame，由 init_service 注入


# ── 初始化 ────────────────────────────────────────────────────
def init_service(df, warmup: bool = True) -> None:
    """
    注入全局 DataFrame，可选是否在启动时预热所有数据。

    Args:
        df:      全局 DataFrame
        warmup:  True = 启动预热（推荐生产环境）；
                 False = 跳过预热，首次访问时按需计算（适合本地快速调试）
    """
    global _df
    _df = df



# ── 查询接口 ──────────────────────────────────────────────────
def get_all_products() -> list[str]:
    """返回所有产品名称列表"""
    return _df["product_name"].unique().tolist()



def get_product_summary(product: str) -> dict | None:
    """
    返回产品结构化摘要（用于 AI 问答上下文 / 页面展示）。
    优先读缓存，缓存未命中时现场计算并写入缓存。

    Returns:
        dict with keys:
            product_name, price, avg_star, total_comments,
            radar, star_distribution,
            main_positive_keywords, main_negative_keywords, main_problem_keywords
        None if product not found
    """
    # 两级缓存（内存 → 磁盘）
    cached = get_product_summary_cache(product)
    if cached:
        return cached

    sub = _df[_df["product_name"] == product]
    if sub.empty:
        return None

    # ── 基础统计 ──────────────────────────────
    avg_star       = round(sub["comment_star"].mean(), 2)
    total_comments = len(sub)
    price          = sub["price"].iloc[0]
    star_dist      = sub["comment_star"].value_counts().to_dict()

    # ── 雷达图各维度得分 ───────────────────────
    radar = get_radar_data(sub)

    # ── 好/差评关键词（从 DataFrame 字段聚合）──
    positive_kws, negative_kws = [], []
    for _, row in sub.iterrows():
        if row.get("positive_keyword", "无") != "无":
            positive_kws.extend(str(row["positive_keyword"]).split(","))
        if row.get("negative_keyword", "无") != "无":
            negative_kws.extend(str(row["negative_keyword"]).split(","))

    summary = {
        "product_name":           product,
        "price":                  price,
        "avg_star":               avg_star,
        "total_comments":         total_comments,
        "radar":                  radar,
        "star_distribution":      star_dist,
        "main_positive_keywords": list(set(positive_kws))[:5],
        "main_negative_keywords": list(set(negative_kws))[:5],
    }

    set_product_summary_cache(product, summary)
    return summary

