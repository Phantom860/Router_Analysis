# 全局缓存字典
_cluster_cache = {}
_product_summary_cache = {}

def get_cluster_cache(product):
    """获取聚类缓存"""
    return _cluster_cache.get(product)

def set_cluster_cache(product, data):
    """设置聚类缓存"""
    _cluster_cache[product] = data

def get_product_summary_cache(product):
    """获取产品摘要缓存"""
    return _product_summary_cache.get(product)

def set_product_summary_cache(product, data):
    """设置产品摘要缓存"""
    _product_summary_cache[product] = data

def clear_cache():
    """清空所有缓存"""
    global _cluster_cache, _product_summary_cache
    _cluster_cache = {}
    _product_summary_cache = {}

def get_cache_info():
    """获取缓存统计信息"""
    return {
        "cluster_cache_size": len(_cluster_cache),
        "product_summary_cache_size": len(_product_summary_cache)
    }