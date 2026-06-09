from flask import Flask, render_template, request, jsonify
from urllib.parse import parse_qs, urlparse
import pandas as pd

from utils import (
    load_data, build_all_wordclouds,
    get_radar_data, get_star_data, get_cluster_data,
    get_all_products, get_product_summary, init_service,
    build_wordcloud, call_doubao
)

app = Flask(__name__)

# 初始化数据
df = load_data()
build_all_wordclouds(df)
init_service(df)  # 初始化产品服务


# 辅助函数
def get_wordcloud_img(product):
    """获取词云图片路径"""
    
    sub = df[df["product_name"] == product]
    comments_series = sub["word_segment"]  # 根据实际列名调整
    return build_wordcloud(comments_series, product)


# ==================== 页面路由 ====================

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
        return render_template(
            "detail.html", product="", all_products=get_all_products(),
            selected="", wordcloud_img="", radar_data={}, star_data={},
            avg_star=None, count=None, price=None
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
    return render_template(
        "wordcloud.html",
        all_products=get_all_products(),
        selected=product,
        img_path=get_wordcloud_img(product) if product else ""
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
        cluster_data=get_cluster_data(sub) if sub is not None else {}
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


@app.route("/report")
def report():
    product = request.args.get("product_name", "")
    if not product:
        return render_template("report.html", all_products=get_all_products(), selected="", report="")
    
    sub = df[df["product_name"] == product]
    comments = "\n".join(sub["clean_comment"].dropna().astype(str).head(30).tolist())
    
    prompt = f"""
请根据以下路由器评价数据生成分析报告：

商品名称：{product}
价格：{sub['price'].iloc[0]}
平均评分：{round(sub['comment_star'].mean(), 2)}
评论内容：{comments}

请输出：
1. 产品整体评价
2. 用户最满意的地方
3. 用户主要抱怨
4. 适用人群
5. 购买建议

控制在500字以内。
"""
    report = call_doubao(prompt)
    return render_template("report.html", all_products=get_all_products(), selected=product, report=report)


# ==================== AI助手接口 ====================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    msg = data.get("msg", "")
    page = data.get("page", "")
    
    if not msg:
        return jsonify({"reply": "请输入问题"})
    
    # 从页面路径识别当前产品
    product = None
    if "/detail/" in page:
        product = page.split("/detail/")[1].split("?")[0]
    elif page in ["/detail", "/report"]:
        parsed = urlparse(page)
        params = parse_qs(parsed.query)
        product = params.get("product_name", [None])[0]
    
    # 如果识别到产品，使用产品摘要回答问题
    if product:
        summary = get_product_summary(product)
        if summary:
            prompt = f"""
用户正在浏览产品【{product}】的页面，问了一个问题，请基于以下数据回答。

【产品数据】
价格：{summary['price']}元
评分：{summary['avg_star']}星
总评论数：{summary['total_comments']}
各维度表现（好评率）：{', '.join([f"{d}:{v}%" for d, v in zip(summary['radar']['dimensions'], summary['radar']['values'])])}
主要好评：{', '.join(summary['main_positive_keywords']) if summary['main_positive_keywords'] else '无'}
主要差评：{', '.join(summary['main_negative_keywords']) if summary['main_negative_keywords'] else '无'}

用户问题：{msg}

请结合数据给出简洁、专业的回答，150字以内。
"""
            reply = call_doubao(prompt, temperature=0.2)
            return jsonify({"reply": reply})
    
    # 没有具体产品时的通用回答
    prompt = f"""
用户正在浏览路由器评价分析系统的页面，当前页面：{page}
用户问：{msg}
请作为采购分析助手，结合路由器选购知识给出回答。控制在150字以内。
"""
    reply = call_doubao(prompt, temperature=0.3)
    return jsonify({"reply": reply})


@app.route("/api/page_summary", methods=["POST"])
def page_summary():
    data = request.get_json()
    page = data.get("page", "")
    
    prompt = f"""
当前用户正在浏览：{page}
这是一个路由器评价分析系统。

请告诉用户：
当前页面主要展示什么内容，
应该如何阅读，
能得到什么结论。

控制在150字以内。
"""
    summary = call_doubao(prompt)
    return jsonify({"summary": summary})


if __name__ == "__main__":
    app.run(debug=True)