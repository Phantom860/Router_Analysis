from flask import Flask, render_template, request, jsonify
import pandas as pd

from utils import (
    load_data, warmup_all,
    get_radar_data, get_star_data, get_cluster_data,
    get_all_products,  init_service,
    chat_answer, generate_report,
    get_wordcloud_url
)

app = Flask(__name__)

# 初始化数据
df = load_data()
warmup_all(df)  # 预生成所有词云图片
init_service(df)  # 初始化产品服务


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

    # 使用统一的函数获取词云URL
    wordcloud_url = get_wordcloud_url(product, df)

    return render_template(
        "detail.html",
        product=product,
        avg_star=round(sub["comment_star"].mean(), 2),
        count=len(sub),
        price=sub["price"].iloc[0],
        all_products=get_all_products(),
        selected=product,
        wordcloud_img=wordcloud_url,  # 使用统一的URL
        radar_data=get_radar_data(sub),
        star_data=get_star_data(sub),
    )


@app.route("/detail/<product>")
def detail(product):
    sub = df[df["product_name"] == product]
    wordcloud_url = get_wordcloud_url(product, df)

    return render_template(
        "detail.html",
        product=product,
        avg_star=round(sub["comment_star"].mean(), 2),
        count=len(sub),
        price=sub["price"].iloc[0],
        all_products=get_all_products(),
        selected=product,
        wordcloud_img=wordcloud_url,
        radar_data=get_radar_data(sub),
        star_data=get_star_data(sub),
    )


@app.route("/wordcloud")
def wordcloud():
    product = request.args.get("product_name", "")
    wordcloud_url = get_wordcloud_url(product, df) if product else ""

    return render_template(
        "wordcloud.html",
        all_products=get_all_products(),
        selected=product,
        img_path=wordcloud_url
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
    cluster_data = get_cluster_data(sub) if sub is not None else {}

    return render_template(
        "cluster.html",
        all_products=get_all_products(),
        selected=product,
        cluster_data=cluster_data,
        ai_summary=cluster_data.get("ai_summary", "")
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



# ==================== AI助手接口 ====================

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

    report_text = generate_report(
        product,
        sub
    )

    return render_template(
        "report.html",
        all_products=get_all_products(),
        selected=product,
        report=report_text
    )


@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json()

    question = data.get("msg", "").strip()
    product = data.get("product")

    if not question:
        return jsonify({
            "success": False,
            "answer": "请输入问题"
        })

    try:
        answer = chat_answer(
            question=question,
            product=product
        )

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "answer": f"AI服务异常：{str(e)}"
        })


if __name__ == "__main__":
    app.run(debug=True)