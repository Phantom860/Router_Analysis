# ============================================================
# Project: Router_Analysis
# File:    router_analyze.py
# Module:  组员B - 数据清洗、分词、情感分析、关键词提取
# Function: 读取原始评价数据 → 清洗 → 分词 → 分析 → 输出标准结果
# Input:   router_raw_data.csv
# Output:  router_analysis_data.csv
# ============================================================

# ------------------- 1. 导入需要的工具库 -------------------
import pandas as pd      # 数据处理
import jieba             # 中文分词
import re                # 文本清洗
import warnings
warnings.filterwarnings("ignore")

# ------------------- 2. 全局配置 -------------------
# 停用词：过滤无意义词汇
stop_words = {
    "的", "了", "和", "是", "在", "我", "有", "就", "都",
    "很", "也", "还", "这个", "那个", "但是", "因为", "所以",
    "可以", "应该", "感觉", "觉得", "用", "买", "安装", "非常"
}

# 优点关键词库
positive_words = [
    "网速快", "信号好", "稳定", "穿墙强", "散热好",
    "覆盖广", "设置简单", "颜值高", "网速稳定"
]

# 缺点关键词库
negative_words = [
    "信号差", "断流", "发热", "穿墙弱", "卡顿",
    "设置复杂", "价格贵", "覆盖小", "经常掉线"
]

# ------------------- 3. 读取原始数据 -------------------
print("正在读取原始数据：router_raw_data.csv")
df = pd.read_csv("router_raw_data.csv", encoding="utf-8")

# ------------------- 4. 数据清洗 -------------------
def clean_comment(text):
    """清洗评论文本：去除符号、表情、广告、乱码"""
    if pd.isna(text):
        return ""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("正在清洗评价数据...")
df["clean_comment"] = df["user_comment"].apply(clean_comment)
df = df[df["clean_comment"] != ""].reset_index(drop=True)

# ------------------- 5. 中文分词 + 过滤停用词 -------------------
def word_segment(text):
    """对清洗后的文本分词，并过滤无意义词语"""
    if text == "":
        return ""
    words = jieba.lcut(text)
    words_filtered = [word for word in words if word not in stop_words and len(word) > 1]
    return " ".join(words_filtered)

print("正在进行中文分词...")
df["word_segment"] = df["clean_comment"].apply(word_segment)

# ------------------- 6. 情感分类（好评/中评/差评） -------------------
def get_sentiment(star):
    """根据星级判断情感类型"""
    if star >= 4:
        return "好评"
    elif star == 3:
        return "中评"
    else:
        return "差评"

print("正在进行情感分类...")
df["sentiment_type"] = df["comment_star"].apply(get_sentiment)

# ------------------- 7. 提取优点关键词 -------------------
def extract_positive(text):
    """提取好评关键词"""
    result = [w for w in positive_words if w in text]
    return ",".join(result) if result else "无"

# ------------------- 8. 提取缺点关键词 -------------------
def extract_negative(text):
    """提取差评关键词（已修复语法错误）"""
    result = [w for w in negative_words if w in text]
    return ",".join(result) if result else "无"

print("正在提取优缺点关键词...")
df["positive_keyword"] = df["clean_comment"].apply(extract_positive)
df["negative_keyword"] = df["clean_comment"].apply(extract_negative)

# ------------------- 9. 提取用户关注点 -------------------
def get_focus(text):
    """提取用户最关心的维度：网速、信号、稳定性、价格、易用性"""
    focus = []
    if "网速" in text or "快" in text:
        focus.append("网速")
    if "信号" in text or "穿墙" in text:
        focus.append("信号")
    if "稳定" in text or "断流" in text:
        focus.append("稳定性")
    if "价格" in text:
        focus.append("价格")
    if "设置" in text:
        focus.append("易用性")
    return ",".join(focus) if focus else "其他"

df["focus_word"] = df["clean_comment"].apply(get_focus)

# ------------------- 10. 输出最终分析结果 -------------------
output_columns = [
    "product_name",
    "brand",
    "price",
    "clean_comment",
    "word_segment",
    "sentiment_type",
    "positive_keyword",
    "negative_keyword",
    "focus_word",
    "comment_star"
]

result_df = df[output_columns]
result_df.to_csv("router_analysis_data.csv", index=False, encoding="utf-8")

# ------------------- 11. 完成提示 -------------------
print("=" * 60)
print("组员B 数据分析完成！")
print("输出文件：router_analysis_data.csv")
print(f"有效评价数量：{len(result_df)} 条")
print("=" * 60)