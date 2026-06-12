import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from .cache import get_cluster_cache, set_cluster_cache
from .ai_service import generate_cluster_summary

def get_cluster_data(sub):
    """
    差评聚类分析（1-2星）

    Args:
        sub: 单个商品的DataFrame

    Returns:
        dict: {"clusters": [...], "raw_comments": [...]}
    """
    product = sub["product_name"].iloc[0] if not sub.empty else ""

    # 检查缓存
    cached = get_cluster_cache(product)
    if cached:
        return cached

    # 获取差评（1-2星）
    neg_df = sub[sub["comment_star"] <= 2].copy()

    if len(neg_df) < 5:
        result = {
            "clusters": [],
            "raw_comments": neg_df["clean_comment"].head(20).tolist(),
            "total_negative": len(neg_df)
        }
        set_cluster_cache(product, result)
        return result

    comments = neg_df["clean_comment"].dropna().astype(str).tolist()

    # 向量化
    vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
    X = vectorizer.fit_transform(comments)

    # 聚类（动态决定类别数）
    n_clusters = min(2, max(1, len(comments) // 3))
    if n_clusters < 2:
        result = {
            "clusters": [],
            "raw_comments": comments[:20],
            "total_negative": len(neg_df)
        }
        set_cluster_cache(product, result)
        return result

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # 提取每类的关键词
    feature_names = vectorizer.get_feature_names_out()
    clusters_info = []

    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-5:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]

        # 取该类中的代表性评论（前2条）
        cluster_indices = np.where(labels == i)[0]
        sample_comments = [comments[j] for j in cluster_indices[:2]]

        clusters_info.append({
            "cluster_id": i,
            "keywords": top_words,
            "examples": sample_comments,
            "size": int((labels == i).sum())
        })

    result = {
        "clusters": clusters_info,
        "raw_comments": comments[:20],
        "total_negative": len(neg_df)
    }

    # === 新增AI总结 ===
    ai_summary = generate_cluster_summary(result)

    result = {
        "clusters": clusters_info,
        "raw_comments": comments[:20],
        "total_negative": len(neg_df),
        "ai_summary": ai_summary
    }
    set_cluster_cache(product, result)
    return result

