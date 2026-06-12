
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

