# utils/wordcloud_builder.py

import re
import os
import jieba
import jieba.posseg as pseg
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# ─────────────────────────────────────────────
#  停用词表（只保留真正通用的虚词/套话）
# ─────────────────────────────────────────────
BASE_STOP_WORDS = {
    # 代词
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '自己', '别人',
    '这个', '那个', '这些', '那些', '这里', '那里', '哪', '每', '各',
    # 虚词
    '的', '了', '是', '在', '和', '与', '就', '都', '也', '还', '只',
    '但', '却', '又', '才', '更', '最', '很', '太', '非常', '特别',
    '有点', '有些', '可以', '能', '会', '要', '想', '觉得', '感觉',
    # 时间/数量
    '一个', '两个', '三个', '昨天', '今天', '明天', '之前', '之后',
    '以前', '以后', '当时', '现在', '已经', '正在', '还要', '就能',
    # 电商套话
    '好评', '差评', '中评', '追评', '晒图', '收货', '已购', '购物', '匿名',
    '京东', '淘宝', '天猫', '拼多多', '快递', '物流', '包装', '客服',
    '用户', '卖家', '商家', '平台', '活动', '未填写', '默认', 
    # 无意义单字
    '这', '那', '哪', '每', '各', '被', '把', '给', '让', '叫',
    '吗', '呢', '啊', '哦', '嗯', '吧', '嘛', '哈', '呀',
    # 无意义动词
    '打算', '收到', '购买', '使用', '感觉', '觉得', '体验', '安装',
      '设置', '连接', '更新', '下载', '取消',
    # 品牌
    '华为', '小米', '中兴', '普联', 'TP-Link', 
    # 物品词
    '路由器', '网络', '手机', '耳机', '充电器', '数据线', '保护套', '屏幕', '电池',
}

# 保留的词性前缀
# n=名词 a=形容词 i=成语 l=习语 nz=其他专有名词 nt=机构名
KEEP_POS = {'n', 'a', 'i', 'l', 'nz', 'nt', 'vn'}
# 过滤掉的词性前缀（遇到就直接丢）
DROP_POS = {'r', 'c', 'u', 'y', 'p', 'e', 'o', 'd', 'xc', 'm', 'q'}

NUMBER_PATTERN = re.compile(r'^\d+[\d.]*$')


# ─────────────────────────────────────────────
#  Step 1: 分词 + 词性过滤（单条评论级别）
# ─────────────────────────────────────────────
def tokenize_with_pos(text: str) -> list[str]:
    """
    对单条评论做分词+词性过滤，返回保留的词列表。
    策略：只要词性前缀在 KEEP_POS 里就保留，在 DROP_POS 里就丢，其余酌情保留。
    """
    result = []
    for word, flag in pseg.cut(text):
        word = word.strip()
        # 长度/数字过滤
        if len(word) < 2:
            continue
        if NUMBER_PATTERN.match(word):
            continue
        if not any(c.isalnum() for c in word):
            continue
        # 停用词
        if word in BASE_STOP_WORDS:
            continue
        # 词性判断
        pos_prefix = flag[:2] if len(flag) >= 2 else flag
        if any(flag.startswith(p) for p in DROP_POS):
            continue
        result.append(word)
    return result


# ─────────────────────────────────────────────
#  Step 2: TF-IDF 筛选高价值词
# ─────────────────────────────────────────────
def get_tfidf_topwords(
    tokenized_docs: list[list[str]],
    top_n: int = 200,
    max_df: float = 0.6,   # 出现在超过60%评论里的词 → 太普通，砍
    min_df: int = 2,        # 至少要在2条评论里出现（过滤低频噪声）
) -> set[str]:
    """
    用 TF-IDF 从所有评论中选出最有区分度的 top_n 个词。

    max_df=0.6 的意思：
        "不错""方便" 这种词几乎每条评论都有 → DF很高 → TF-IDF自动压低 → 排名靠后 → 被cut掉

    返回值是词的集合，后续用来做白名单过滤。
    """
    # 把每条评论的词列表拼成空格分隔的字符串（TfidfVectorizer 需要这个格式）
    corpus = [" ".join(doc) for doc in tokenized_docs if doc]
    if len(corpus) < 2:
        # 评论太少，TF-IDF没意义，直接返回所有词
        return set(w for doc in tokenized_docs for w in doc)

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\S+\b",  # 匹配中文词（已经是空格分隔）
        max_df=max_df,
        min_df=min_df,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return set(w for doc in tokenized_docs for w in doc)

    feature_names = vectorizer.get_feature_names_out()
    # 每个词取其在所有文档里的最大 TF-IDF 值（代表它"最重要的那一刻"）
    scores = tfidf_matrix.max(axis=0).toarray().flatten()
    # 按分数降序，取前 top_n 个
    top_indices = np.argsort(scores)[::-1][:top_n]
    top_words = set(feature_names[i] for i in top_indices)
    return top_words


# ─────────────────────────────────────────────
#  Step 3: 统计词频（用于词云权重）
# ─────────────────────────────────────────────
def count_words(tokenized_docs: list[list[str]], whitelist: set[str]) -> dict[str, int]:
    """统计通过白名单过滤后的词频，词云大小按词频显示。"""
    freq = {}
    for doc in tokenized_docs:
        for word in doc:
            if word in whitelist:
                freq[word] = freq.get(word, 0) + 1
    return freq


# ─────────────────────────────────────────────
#  主函数：生成词云
# ─────────────────────────────────────────────
def build_wordcloud(
    comments_series,        # pd.Series，每个元素是一条评论文本
    product_name: str,
    force_rebuild: bool = False,
    top_n: int = 150,       # TF-IDF 保留词数
    max_df: float = 0.6,    # DF 上限
    min_df: int = 2,
) -> str:
    """
    生成词云图片，返回图片 URL 路径。

    流水线：
        原始评论
          → POS 分词过滤（去代词/虚词/副词）
          → TF-IDF 选高价值词（去普通词/高频套话）
          → 词频统计 → 词云
    """
    os.makedirs("static/wordcloud", exist_ok=True)
    filename = f"{product_name}.png"
    save_path = os.path.join("static/wordcloud", filename)
    url_path = f"/static/wordcloud/{filename}"

    if os.path.exists(save_path) and not force_rebuild:
        return url_path

    # 清洗 + 分词
    comments = comments_series.dropna().astype(str).tolist()
    if not comments:
        return ""

    print(f"[词云] {product_name}：共 {len(comments)} 条评论，开始处理...")
    tokenized = [tokenize_with_pos(c) for c in comments]
    total_tokens = sum(len(d) for d in tokenized)
    print(f"[词云] POS 过滤后共 {total_tokens} 个词")

    # TF-IDF 选词
    whitelist = get_tfidf_topwords(tokenized, top_n=top_n, max_df=max_df, min_df=min_df)
    print(f"[词云] TF-IDF 白名单词数：{len(whitelist)}")
    print(f"[词云] Top词样例：{list(whitelist)[:20]}")

    # 统计词频
    freq = count_words(tokenized, whitelist)
    if not freq:
        print(f"[词云] 警告：{product_name} 过滤后无有效词汇，尝试放宽 max_df")
        return ""

    # 生成词云（直接传词频字典，比传文本更精准）
    wc = WordCloud(
        font_path="simhei.ttf",
        background_color="white",
        width=1200,
        height=700,
        max_words=100,
        collocations=False,
    )
    wc.generate_from_frequencies(freq)
    wc.to_file(save_path)
    print(f"[词云] 已保存至 {save_path}")
    return url_path


def get_wordcloud_url(product_name: str, df, force_rebuild: bool = False) -> str:
    """Flask 路由调用的便捷入口，接口与原代码一致。"""
    sub = df[df["product_name"] == product_name]
    if sub.empty:
        return ""
    return build_wordcloud(sub["clean_comment"], product_name, force_rebuild)

