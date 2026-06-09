# utils/wordcloud_builder.py

import re
import jieba.posseg as pseg  # 词性标注，需要: pip install jieba
from wordcloud import WordCloud 
import os

class SmartWordCloudFilter:
    """智能词云过滤器"""
    
    # 基础停用词（通用，不会变）
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
        
        # 评论套话
        '好评', '差评', '中评', '追评', '晒图', '收货', '已购', '购物',
        '京东', '淘宝', '天猫', '拼多多', '快递', '物流', '包装',
        
        # 无意义单字
        '这', '那', '哪', '每', '各', '被', '把', '给', '让', '叫',
        '吗', '呢', '啊', '哦', '嗯', '吧',
    }
    
    # 长度规则
    MIN_WORD_LEN = 2          # 中文至少2字
    MIN_EN_WORD_LEN = 3       # 英文至少3字母
    
    # 数字模式
    NUMBER_PATTERN = re.compile(r'^\d+$')
    DATE_PATTERN = re.compile(r'^\d{4,8}$')
    
    @classmethod
    def is_valid_word(cls, word):
        """判断一个词是否应该保留"""
        
        # 1. 长度过滤
        if len(word) < cls.MIN_WORD_LEN:
            return False
        if word.isalpha() and len(word) < cls.MIN_EN_WORD_LEN:
            return False
        
        # 2. 纯数字过滤
        if cls.NUMBER_PATTERN.match(word):
            return False
        
        # 3. 日期过滤
        if cls.DATE_PATTERN.match(word) and word.isdigit():
            return False
        
        # 4. 基础停用词
        if word in cls.BASE_STOP_WORDS:
            return False
        
        # 5. 全是标点符号
        if all(not c.isalnum() for c in word):
            return False
        
        return True
    
    @classmethod
    def is_meaningful_noun(cls, word):
        """判断是否为有意义的实词（名词/形容词/动词）"""
        try:
            # 词性标注：n=名词, a=形容词, v=动词, d=副词, ns=地名
            words = pseg.cut(word)
            for w, flag in words:
                if w == word:
                    # 保留名词、形容词、动词、品牌词
                    if flag.startswith(('n', 'a', 'v', 'nz', 'nt')):
                        return True
                    # 但过滤掉代词(r)、连词(c)、助词(u)、语气词(y)
                    if flag.startswith(('r', 'c', 'u', 'y')):
                        return False
        except:
            pass
        return True
    
    @classmethod
    def clean_text(cls, text, strict=False):
        """
        清洗文本
        
        Args:
            text: 空格分隔的字符串
            strict: True=只保留有意义的实词, False=保留所有有效词
        """
        words = text.split()
        result = []
        
        for w in words:
            # 基础过滤
            if not cls.is_valid_word(w):
                continue
            
            # 可选：严格模式，只保留实词
            if strict and not cls.is_meaningful_noun(w):
                continue
            
            result.append(w)
        
        return " ".join(result)


def build_wordcloud(comments_series, product_name, force_rebuild=False, strict_mode=False):
    """生成词云"""
    os.makedirs("static/wordcloud", exist_ok=True)
    
    filename = f"{product_name}.png"
    save_path = os.path.join("static/wordcloud", filename)
    url_path = f"/static/wordcloud/{filename}"
    
    if os.path.exists(save_path) and not force_rebuild:
        return url_path
    
    all_words = " ".join(comments_series.dropna().astype(str))
    if not all_words.strip():
        return ""
    
    # 智能清洗
    clean_text = SmartWordCloudFilter.clean_text(all_words, strict=strict_mode)
    
    if not clean_text.strip():
        print(f"警告：{product_name} 清洗后无有效词汇")
        return ""
    
    # 生成词云
    wc = WordCloud(
        font_path="simhei.ttf",
        background_color="white",
        width=1200,
        height=700,
        max_words=100,
        collocations=False,  # 不显示词组，只显示单个词
    )
    wc.generate(clean_text)
    wc.to_file(save_path)
    
    return url_path


def get_wordcloud_url(product_name, df, force_rebuild=False):
    """
    获取商品词云图片URL的便捷函数
    
    Args:
        product_name: 商品名称
        df: 包含评论数据的DataFrame
        force_rebuild: 是否强制重新生成
    
    Returns:
        str: 词云图片的URL路径
    """
    sub = df[df["product_name"] == product_name]
    if sub.empty:
        return ""
    
    return build_wordcloud(sub["word_segment"], product_name, force_rebuild)