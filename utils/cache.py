"""
cache.py —— 两级缓存：内存（本次进程）+ 磁盘（JSON持久化）

目录结构：
  static/cache/cluster/   存各产品聚类结果
  static/cache/summary/   存各产品摘要

重启后先读磁盘，命中则直接返回，不再重新计算/调用AI。
调用方不感知两级细节，接口与原来完全一致。
"""

import os
import json
import hashlib

# ── 磁盘缓存根目录 ──────────────────────────────────────────
CACHE_DIR_CLUSTER = os.path.join("static", "cache", "cluster")
CACHE_DIR_SUMMARY = os.path.join("static", "cache", "summary")
CACHE_DIR_REPORT  = os.path.join("static", "cache", "report")

os.makedirs(CACHE_DIR_CLUSTER, exist_ok=True)
os.makedirs(CACHE_DIR_SUMMARY, exist_ok=True)
os.makedirs(CACHE_DIR_REPORT,  exist_ok=True)

# ── 内存缓存（进程内加速，避免重复 I/O）───────────────────────
_cluster_cache: dict = {}
_product_summary_cache: dict = {}
_report_cache: dict = {}


# ── 工具函数 ────────────────────────────────────────────────
def _safe_filename(name: str) -> str:
    """把产品名转成安全文件名（含中文的名字直接md5）"""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    if len(safe) > 60 or safe != name:
        safe = hashlib.md5(name.encode()).hexdigest()
    return safe + ".json"


def _disk_path(cache_dir: str, product: str) -> str:
    return os.path.join(cache_dir, _safe_filename(product))


def _read_disk(cache_dir: str, product: str):
    path = _disk_path(cache_dir, product)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_disk(cache_dir: str, product: str, data) -> None:
    path = _disk_path(cache_dir, product)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[缓存] 磁盘写入失败 {path}：{e}")


# ── 聚类缓存 ────────────────────────────────────────────────
def get_cluster_cache(product: str):
    """先查内存，再查磁盘，都没有返回 None"""
    if product in _cluster_cache:
        return _cluster_cache[product]
    data = _read_disk(CACHE_DIR_CLUSTER, product)
    if data is not None:
        _cluster_cache[product] = data   # 回填内存
    return data


def set_cluster_cache(product: str, data) -> None:
    """同时写内存和磁盘"""
    _cluster_cache[product] = data
    _write_disk(CACHE_DIR_CLUSTER, product, data)


# ── 产品摘要缓存 ─────────────────────────────────────────────
def get_product_summary_cache(product: str):
    if product in _product_summary_cache:
        return _product_summary_cache[product]
    data = _read_disk(CACHE_DIR_SUMMARY, product)
    if data is not None:
        _product_summary_cache[product] = data
    return data


def set_product_summary_cache(product: str, data) -> None:
    _product_summary_cache[product] = data
    _write_disk(CACHE_DIR_SUMMARY, product, data)


# ── Report 总结缓存 ───────────────────────────────────────────
def get_report_cache(product: str) -> str | None:
    """先查内存，再查磁盘，都没有返回 None"""
    if product in _report_cache:
        return _report_cache[product]
    data = _read_disk(CACHE_DIR_REPORT, product)
    if data is not None:
        _report_cache[product] = data
    return data


def set_report_cache(product: str, report_text: str) -> None:
    """同时写内存和磁盘"""
    _report_cache[product] = report_text
    _write_disk(CACHE_DIR_REPORT, product, report_text)


# ── 清空缓存 ─────────────────────────────────────────────────
def clear_cache(disk: bool = True) -> None:
    """
    清空所有缓存。
    disk=True 时同时删除磁盘文件，False 时只清内存（用于调试）。
    """
    global _cluster_cache, _product_summary_cache, _report_cache
    _cluster_cache = {}
    _product_summary_cache = {}
    _report_cache = {}

    if disk:
        for cache_dir in [CACHE_DIR_CLUSTER, CACHE_DIR_SUMMARY, CACHE_DIR_REPORT]:
            for fname in os.listdir(cache_dir):
                if fname.endswith(".json"):
                    try:
                        os.remove(os.path.join(cache_dir, fname))
                    except Exception:
                        pass
        print("[缓存] 磁盘缓存已清空")


def get_cache_info() -> dict:
    """返回内存缓存统计（不扫描磁盘）"""
    return {
        "cluster_mem":  len(_cluster_cache),
        "summary_mem":  len(_product_summary_cache),
        "report_mem":   len(_report_cache),
        "cluster_disk": len(os.listdir(CACHE_DIR_CLUSTER)),
        "summary_disk": len(os.listdir(CACHE_DIR_SUMMARY)),
        "report_disk":  len(os.listdir(CACHE_DIR_REPORT)),
    }