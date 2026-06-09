from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
from utils.preprocess import (load_data, build_all_wordclouds)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)
df = load_data()
build_all_wordclouds(df)

# 加载环境变量
load_dotenv()

# 全局缓存字典
_cluster_cache = {}

# 火山配置
AK = os.getenv("VOLC_AK")
SK = os.getenv("VOLC_SK")
MODEL_ID = os.getenv("MODEL_ID")
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"


# ══════════════════════════════════════════
#  公共辅助函数
# ══════════════════════════════════════════

def get_all_products():
    return df["product_name"].unique().tolist()


def get_wordcloud_img(product):

    filename = f"static/wordcloud/{product}.png"

    if os.path.exists(filename):

        return "/" + filename

    return ""


def get_radar_data(sub):
    """返回雷达图所需数据，值为各维度的好评率（0-100）"""
    # 维度与关键词映射（可基于 focus_word 或直接匹配评论内容）
    dimension_keywords = {
        "网速": ["网速", "快", "慢", "延迟", "下载"],
        "稳定性": ["稳定", "掉线", "断流", "卡顿"],
        "覆盖范围": ["信号", "穿墙", "覆盖", "满格"],
        "发热": ["发热", "烫", "温度"],
        "性价比": ["价格", "便宜", "贵", "划算", "值"]
    }

    # 情感分值映射
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
    """统计各星级数量，返回柱状图所需数据"""
    counts = sub["comment_star"].value_counts().sort_index()
    return {
        "stars":  counts.index.astype(int).tolist(),
        "counts": counts.values.tolist()
    }


def get_cluster_data(sub):
    product = sub["product_name"].iloc[0] if not sub.empty else ""
    if product in _cluster_cache:
        return _cluster_cache[product]

    # 获取差评（1-2星）的清洁评论
    neg_df = sub[sub["comment_star"] <= 2].copy()
    if len(neg_df) < 5:
        result = {"clusters": [], "raw_comments": neg_df["clean_comment"].head(20).tolist()}
        _cluster_cache[product] = result
        return result

    comments = neg_df["clean_comment"].dropna().astype(str).tolist()

    # 向量化
    vectorizer = TfidfVectorizer(max_features=100, stop_words=None)  # 停用词已在清洗中处理
    X = vectorizer.fit_transform(comments)

    # 聚类（假设分为2类）
    n_clusters = min(2, len(comments) // 3)
    if n_clusters < 2:
        result = {"clusters": [], "raw_comments": comments[:20]}
        _cluster_cache[product] = result
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
        sample_comments = [comments[j] for j in np.where(labels == i)[0][:2]]
        clusters_info.append({
            "cluster_id": i,
            "keywords": top_words,
            "examples": sample_comments,
            "size": int((labels == i).sum())
        })

    result = {"clusters": clusters_info, "raw_comments": comments[:20]}
    _cluster_cache[product] = result
    return result


# 请求火山方舟接口函数
def call_doubao(prompt_text):
    headers = {
        "Authorization": f"Bearer {AK}/{SK}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "你是电信内部WiFi路由器采购分析助手，根据路由器参数、电商评价做选购分析、对比总结、输出导购建议，回答专业简洁，可输出表格对比优缺点。"
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "temperature": 0.3,  # 越低回答越严谨稳定，采购场景推荐0.2~0.4
        "max_tokens": 2000
    }
    try:
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        res_json = resp.json()
        if res_json.get("choices"):
            return res_json["choices"][0]["message"]["content"]
        else:
            return f"模型调用失败：{res_json}"
    except Exception as e:
        return f"接口异常：{str(e)}"



# ══════════════════════════════════════════
#  路由
# ══════════════════════════════════════════

@app.route("/")
def index():
    products = (
        df.groupby("product_name")
        .agg({"price": "first", "comment_star": "mean", "clean_comment": "count"})
        .rename(columns={"clean_comment": "user_comment"})
        .reset_index()
    )
    return render_template("index.html", products=products.to_dict("records"))


@app.route("/detail")
def detail_select():
    product = request.args.get("product_name", "")
    if not product:
        # 没选商品，只渲染选择器，不传图表数据
        return render_template(
            "detail.html",
            product="",
            all_products=get_all_products(),
            selected="",
            wordcloud_img="",
            radar_data={},
            star_data={},
            avg_star=None,
            count=None,
            price=None,
        )
    sub = df[df["product_name"] == product]
    return render_template(
        "detail.html",
        product=product,
        avg_star=round(sub["comment_star"].mean(), 2),
        count=len(sub),
        price=sub["price"].iloc[0],
        all_products=get_all_products(),
        selected=product,
        wordcloud_img=get_wordcloud_img(product),
        radar_data=get_radar_data(sub),
        star_data=get_star_data(sub),
    )

@app.route("/detail/<product>")
def detail(product):
    sub = df[df["product_name"] == product]
    return render_template(
        "detail.html",
        product=product,
        avg_star=round(sub["comment_star"].mean(), 2),
        count=len(sub),
        price=sub["price"].iloc[0],
        all_products=get_all_products(),
        selected=product,
        wordcloud_img=get_wordcloud_img(product),
        radar_data=get_radar_data(sub),
        star_data=get_star_data(sub),
    )


@app.route("/wordcloud")
def wordcloud():
    product = request.args.get("product_name", "")
    sub = df[df["product_name"] == product] if product else None
    return render_template(
        "wordcloud.html",
        all_products=get_all_products(),
        selected=product,
        img_path=get_wordcloud_img(product) if sub is not None else ""
    )


@app.route("/radar")
def radar():
    product = request.args.get("product_name", "")
    sub = df[df["product_name"] == product] if product else None
    return render_template(
        "radar.html",
        all_products=get_all_products(),
        selected=product,
        radar_data=get_radar_data(sub) if sub is not None else {}
    )


@app.route("/cluster")
def cluster():
    product = request.args.get("product_name", "")
    sub = df[df["product_name"] == product] if product else None
    return render_template(
        "cluster.html",
        all_products=get_all_products(),
        selected=product,
        cluster_data=get_cluster_data(sub) if sub is not None else []
    )



@app.route("/star")
def star():
    product = request.args.get("product_name", "")
    sub = df[df["product_name"] == product] if product else None
    return render_template(
        "star.html",
        all_products=get_all_products(),
        selected=product,
        star_data=get_star_data(sub) if sub is not None else {}
    )


# AI报告分析
@app.route("/report")
def report():

    product = request.args.get("product_name", "")

    if not product:
        return render_template(
            "report.html",
            all_products=get_all_products(),
            selected="",
            report=""
        )

    sub = df[df["product_name"] == product]

    comments = "\n".join(
        sub["clean_comment"]
        .dropna()
        .astype(str)
        .head(30)
        .tolist()
    )

    prompt = f"""
请根据以下路由器评价数据生成分析报告：

商品名称：
{product}

价格：
{sub["price"].iloc[0]}

平均评分：
{round(sub["comment_star"].mean(),2)}

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

    report = call_doubao(prompt)

    return render_template(
        "report.html",
        all_products=get_all_products(),
        selected=product,
        report=report
    )


# 页面总结接口
@app.route(
"/api/page_summary",
methods=["POST"]
)
def page_summary():

    data=request.get_json()

    page=data.get("page","")

    prompt=f"""
当前用户正在浏览：

{page}

这是一个路由器评价分析系统。

请告诉用户：

当前页面主要展示什么内容，
应该如何阅读，
能得到什么结论。

控制在150字以内。
"""

    summary = call_doubao(prompt)

    return jsonify({
        "summary":summary
    })


if __name__ == "__main__":
    app.run(debug=True)