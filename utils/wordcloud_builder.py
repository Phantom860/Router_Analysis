from wordcloud import WordCloud
import os

def build_wordcloud(comments_series, product):
    """
    comments_series: pandas Series，内容为 word_segment 列（空格分隔的清洗后词汇）
    """
    save_path = f"static/wordcloud/{product}.png"
    if os.path.exists(save_path):
        return save_path

    # 将所有评论的分词结果合并为一个长字符串
    all_words = " ".join(comments_series.dropna().astype(str))
    if not all_words.strip():
        print(f"警告：{product} 没有有效的分词数据")
        return ""

    wc = WordCloud(
        font_path="simhei.ttf",          # 确保字体文件存在
        background_color="white",
        width=1200,
        height=700,
        max_words=200
    )
    wc.generate(all_words)

    os.makedirs("static/wordcloud", exist_ok=True)
    wc.to_file(save_path)
    return save_path