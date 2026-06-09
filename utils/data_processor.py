import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from .cache import get_cluster_cache, set_cluster_cache



def get_radar_data(sub):
    """
    使用滑动窗口共现 + 映射词典计算维度得分（0-100）
    """
    import jieba

    # 维度关键词
    dimension_keywords = {
        "网速": ["网速", "下载速度", "上传速度", "延迟"],
        "稳定性": ["稳定", "掉线", "断流", "卡顿"],
        "覆盖范围": ["信号", "覆盖", "穿墙"],
        "散热表现": ["发热", "温度"],
        "性价比": ["价格", "划算", "便宜", "贵"]
    }

    # 评价词映射表（方向统一，高分越好）
    sentiment_dict = {
        "极快": 100, "很快": 90, "快": 80, "一般": 60, "慢": 30, "很慢": 20,
        "稳定": 95, "不稳定": 30, "掉线": 20, "卡顿": 20,
        "满格": 100, "信号强": 90, "覆盖好": 90, "覆盖差": 20, "弱": 30,
        "不发热": 100, "温度正常": 90, "微热": 70, "发热": 50, "很烫": 20,
        "划算": 95, "值得": 90, "便宜": 85, "贵": 30, "太贵": 10
    }

    window_size = 3  # 滑动窗口大小
    dim_scores_list = {dim: [] for dim in dimension_keywords}

    for comment in sub["clean_comment"]:
        if not comment or str(comment).strip() == "":
            continue

        words = list(jieba.cut(str(comment)))

        for i, word in enumerate(words):
            # 判断是否为属性词
            for dim, keywords in dimension_keywords.items():
                if word in keywords:
                    # 滑动窗口：前后各window_size个词
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    context = words[start:end]

                    # 在窗口中寻找评价词
                    for w in context:
                        if w in sentiment_dict:
                            dim_scores_list[dim].append(sentiment_dict[w])

    # 聚合统计，平均分，未命中维度默认50
    final_scores = []
    for dim in dimension_keywords.keys():
        if dim_scores_list[dim]:
            avg_score = sum(dim_scores_list[dim]) / len(dim_scores_list[dim])
            final_scores.append(round(avg_score, 2))
        else:
            final_scores.append(50)

    return {"dimensions": list(dimension_keywords.keys()), "values": final_scores}


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

    # === 新增AI总结 ===
    ai_summary = summarize_clusters_ai(result, product)

    result = {
        "clusters": clusters_info,
        "raw_comments": comments[:20],
        "total_negative": len(neg_df),
        "ai_summary": ai_summary
    }
    set_cluster_cache(product, result)
    return result


def summarize_clusters_ai(cluster_data, product_name):
    """
    用AI总结聚类结果
    """
    from utils import call_doubao  # 如果已经在同文件可去掉

    clusters = cluster_data.get("clusters", [])
    if not clusters:
        return "差评数据不足，无法生成聚类总结。"

    prompt = f"""
你是一个电商评论分析专家。

请根据以下“差评聚类结果”，总结每一类用户主要在抱怨什么问题。

商品：{product_name}

要求：
- 每一类写1-2句话总结
- 要抓住本质问题，而不是重复关键词
- 输出结构清晰

聚类数据如下：

"""

    for cl in clusters:
        prompt += f"""
【类别 {cl['cluster_id']}】
关键词：{', '.join(cl['keywords'])}
示例评论：{'；'.join(cl['examples'])}
数量：{cl['size']}
"""

    prompt += "\n\n请按类别输出分析结果。"

    return call_doubao(prompt, temperature=0.3)