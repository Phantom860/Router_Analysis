# utils/ai_service.py

from .ai_client import call_doubao
from .product_service import get_product_summary


# =========================
# AI问答
# =========================

def chat_answer(question, product=None):
    """
    AI助手问答
    """

    context = ""

    if product:
        summary = get_product_summary(product)
        if summary:
            context = f"""
商品名称：{summary['product_name']}
价格：{summary['price']}
平均评分：{summary['avg_star']}
评论数：{summary['total_comments']}

优点：
{','.join(summary['main_positive_keywords'])}

缺点：
{','.join(summary['main_negative_keywords'])}

主要问题：
{','.join(summary['main_problem_keywords'])}
"""

    prompt = f"""
用户正在使用路由器评价分析系统。

商品信息：
{context}

用户问题：
{question}

请以采购分析助手身份回答。

要求：
1. 专业
2. 简洁
3. 控制150字以内
"""

    return call_doubao(prompt, temperature=0.3)


# =========================
# AI报告
# =========================

def generate_report(product, sub):
    comments = "\n".join(
        sub["clean_comment"]
        .dropna()
        .astype(str)
        .head(30)
        .tolist()
    )

    prompt = f"""
请根据以下路由器评价数据生成分析报告：

商品名称：{product}
价格：{sub['price'].iloc[0]}
平均评分：{round(sub['comment_star'].mean(),2)}

评论内容：
{comments}

请输出：

1. 产品整体评价
2. 用户最满意的地方
3. 用户主要抱怨
4. 适用人群
5. 购买建议

控制在500字以内。
"""

    return call_doubao(prompt)


# =========================
# 聚类总结
# =========================

def generate_cluster_summary(cluster_data):
    clusters = cluster_data.get("clusters", [])

    if not clusters:
        return "差评数据不足，无法生成聚类总结。"

    text = ""

    for c in clusters:
        text += f"""
类别：{c.get('label','')}
关键词：{','.join(c.get('keywords',[]))}
示例评论：{'；'.join(c.get('examples', []))}
评论数：{c.get('count',0)}
"""

    prompt = f"""
以下是路由器差评聚类结果：

{text}

请总结：

1. 用户最关注的问题
2. 最严重的问题
3. 产品改进建议

控制在200字以内。
"""

    return call_doubao(prompt)