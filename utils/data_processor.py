import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from .cache import get_cluster_cache, set_cluster_cache

def get_radar_data(sub):
    """
    返回雷达图所需数据，值为各维度的好评率（0-100）
    
    Args:
        sub: 单个商品的DataFrame
    
    Returns:
        dict: {"dimensions": [...], "values": [...]}
    """
    dimension_keywords = {
        "网速": ["网速", "快", "慢", "延迟", "下载"],
        "稳定性": ["稳定", "掉线", "断流", "卡顿"],
        "覆盖范围": ["信号", "穿墙", "覆盖", "满格"],
        "发热": ["发热", "烫", "温度"],
        "性价比": ["价格", "便宜", "贵", "划算", "值"]
    }
    
    sentiment_score = {"好评": 100, "中评": 50, "差评": 0}
    
    scores = []
    for dim, keywords in dimension_keywords.items():
        # 找出评论中包含任一关键词的评论
        mask = sub["clean_comment"].apply(
            lambda x: any(kw in str(x) for kw in keywords)
        )
        hit_comments = sub[mask]
        
        if len(hit_comments) == 0:
            scores.append(50)  # 无数据时给中性分
        else:
            avg_score = hit_comments["sentiment_type"].map(sentiment_score).mean()
            scores.append(round(avg_score, 2))
    
    return {"dimensions": list(dimension_keywords.keys()), "values": scores}

def get_star_data(sub):
    """
    统计各星级数量，返回柱状图所需数据
    
    Args:
        sub: 单个商品的DataFrame
    
    Returns:
        dict: {"stars": [...], "counts": [...]}
    """
    counts = sub["comment_star"].value_counts().sort_index()
    return {
        "stars": counts.index.astype(int).tolist(),
        "counts": counts.values.tolist()
    }

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
    set_cluster_cache(product, result)
    return result

def get_wordcloud_img_path(product):
    """获取词云图片路径"""
    import os
    filename = f"static/wordcloud/{product}.png"
    return "/" + filename if os.path.exists(filename) else ""