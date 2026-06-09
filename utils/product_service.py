import pandas as pd
from .cache import get_product_summary_cache, set_product_summary_cache
from .data_processor import get_radar_data, get_cluster_data

# 全局DataFrame
_df = None

def init_service(df):
    """初始化产品服务（传入全局DataFrame）"""
    global _df
    _df = df

def get_all_products():
    """获取所有商品名称列表"""
    return _df["product_name"].unique().tolist()

def get_product_summary(product):
    """
    获取产品分析摘要（用于AI问答）
    
    Args:
        product: 商品名称
    
    Returns:
        dict: 产品摘要信息，如果不存在则返回None
    """
    # 检查缓存
    cached = get_product_summary_cache(product)
    if cached:
        return cached
    
    sub = _df[_df["product_name"] == product]
    if sub.empty:
        return None
    
    # 基础统计
    avg_star = round(sub["comment_star"].mean(), 2)
    total_comments = len(sub)
    price = sub["price"].iloc[0]
    
    # 雷达图数据
    radar = get_radar_data(sub)
    
    # 星级分布
    star_dist = sub["comment_star"].value_counts().to_dict()
    
    # 聚类摘要（差评聚类关键词）
    cluster_data = get_cluster_data(sub)
    problem_keywords = []
    for c in cluster_data.get("clusters", []):
        problem_keywords.extend(c.get("keywords", []))
    problem_keywords = list(set(problem_keywords))[:5]
    
    # 好评/差评关键词
    positive_keywords = []
    negative_keywords = []
    for _, row in sub.iterrows():
        if row["positive_keyword"] != "无":
            positive_keywords.extend(row["positive_keyword"].split(","))
        if row["negative_keyword"] != "无":
            negative_keywords.extend(row["negative_keyword"].split(","))
    
    positive_keywords = list(set(positive_keywords))[:5]
    negative_keywords = list(set(negative_keywords))[:5]
    
    # 构建结构化摘要
    summary = {
        "product_name": product,
        "price": price,
        "avg_star": avg_star,
        "total_comments": total_comments,
        "radar": radar,
        "star_distribution": star_dist,
        "main_positive_keywords": positive_keywords,
        "main_negative_keywords": negative_keywords,
        "main_problem_keywords": problem_keywords,
    }
    
    set_product_summary_cache(product, summary)
    return summary

def get_product_basic_info(product):
    """获取产品基础信息（用于页面显示）"""
    sub = _df[_df["product_name"] == product]
    if sub.empty:
        return None
    
    return {
        "product_name": product,
        "price": sub["price"].iloc[0],
        "avg_star": round(sub["comment_star"].mean(), 2),
        "comment_count": len(sub)
    }