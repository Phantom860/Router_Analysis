"""
Tmall crawler for Group A: collect WiFi6 router product info and comments.

Use links in product_urls.csv:
    python crawler.py --real --max-comments 40 --login-wait 60

Discover products by keyword, then crawl:
    python crawler.py --discover --keyword "WiFi6 router" --product-count 5 --max-comments 20 --login-wait 60
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote, urlparse


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "router_raw_data.csv"
PRODUCT_URLS_FILE = BASE_DIR / "product_urls.csv"
DISCOVERED_URLS_FILE = BASE_DIR / "discovered_product_urls.csv"

CSV_COLUMNS = [
    "product_name",
    "brand",
    "price",
    "wifi_version",
    "net_port",
    "comment_star",
    "user_comment",
    "collect_time",
]

BRAND_KEYWORDS = ["小米", "华为", "TP-LINK", "普联", "中兴", "锐捷", "腾达", "水星", "荣耀", "华硕", "H3C"]
PRODUCT_HINTS = ["WiFi6", "wifi6", "WIFI6", "千兆", "路由", "路由器", "2.5G"]
COMMENT_HINTS = [
    "网速",
    "网络",
    "信号",
    "穿墙",
    "稳定",
    "断网",
    "掉线",
    "安装",
    "设置",
    "路由",
    "速度",
    "延迟",
    "覆盖",
    "连接",
    "好用",
    "不错",
    "满意",
    "一般",
    "差",
    "发热",
    "散热",
    "外观",
    "性价比",
    "宽带",
    "千兆",
    "wifi",
    "WiFi",
]

# 排除提示词
INVALID_COMMENT_HINTS = [
    "发货时间",
    "付款后",
    "小时内发货",
    "七天无理由",
    "运费险",
    "规格参数",
    "商品参数",
    "商品详情",
    "店铺",
    "客服",
    "物流",
    "收藏",
    "加入购物车",
    "立即购买",
    "月销量",
    "优惠券",
    "售后",
    "配送",
    "库存",
]

COMMENT_SELECTORS = [
    "[class*='rate-content']",
    "[class*='RateContent']",
    "[class*='comment']",
    "[class*='Comment']",
    "[class*='review']",
    "[class*='Review']",
    "[class*='content']",
]

REVIEW_GROUPS = [
    ("全部", "null"),
    ("好评", "5"),
    ("中评", "3"),
    ("差评", "1"),
    ("追评", "null"),
]

# 获取当前系统时间（用于记录采集时间）
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 清洗文本内容，统一空值和格式
def text(value: object) -> str:
    result = "" if value is None else str(value).strip()
    result = re.sub(r"\s+", " ", result)
    return result if result else "null"

# 标准化价格格式
def price(value: object) -> str:
    match = re.search(r"\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    if not match:
        return "null"
    try:
        return f"{Decimal(match.group(0)):.2f}"
    except InvalidOperation:
        return "null"

# 评分范围
def star(value: object) -> str:
    try:
        number = int(float(str(value).strip()))
    except (ValueError, AttributeError):
        return "null"
    return str(min(max(number, 1), 5))

# 识别商品支持的 WiFi 协议版本
def wifi_version(value: object, default: str = "WiFi6") -> str:
    haystack = text(value).upper().replace(" ", "")
    if "WIFI6E" in haystack:
        return "WiFi6E"
    if "WIFI6" in haystack:
        return "WiFi6"
    return default

# 品牌
def brand(value: object, default: object = "null") -> str:
    haystack = text(value).upper()
    for keyword in BRAND_KEYWORDS:
        if keyword.upper() in haystack:
            return "TP-LINK" if keyword == "普联" else keyword
    return text(default)

# 网络接口
def net_port(value: object, default: object = "null") -> str:
    haystack = text(value).upper()
    if "2.5G" in haystack or "2500M" in haystack:
        return "2.5G 网口"
    if "千兆" in haystack or "1000M" in haystack or "G口" in haystack:
        return "全千兆网口"
    return text(default)

# 用户评分
def infer_star(comment: str, fallback: str = "null") -> str:
    if any(word in comment for word in ["差评", "失望", "不好", "掉线", "断网", "不满意", "很差"]):
        return "1"
    if any(word in comment for word in ["一般", "还行", "中评", "普通"]):
        return "3"
    if fallback != "null":
        return fallback
    if any(word in comment for word in ["不错", "满意", "好用", "稳定", "很快"]):
        return "5"
    return "4"

# 评论是否有效
def is_valid_comment(comment: str) -> bool:
    if comment == "null" or len(comment) < 8 or len(comment) > 180:
        return False
    if any(word in comment for word in INVALID_COMMENT_HINTS):
        return False
    return any(word in comment for word in COMMENT_HINTS)

# 构造最终保存到CSV中的数据记录
def make_row(product_name: str, brand_name: str, product_price: str, version: str, port: str, score: str, comment: str) -> dict[str, str]:
    return {
        "product_name": text(product_name),
        "brand": text(brand_name),
        "price": price(product_price),
        "wifi_version": wifi_version(version),
        "net_port": text(port),
        "comment_star": star(score),
        "user_comment": text(comment),
        "collect_time": now(),
    }

# 从product_url.csv中读取商品链接
def read_products(path: Path = PRODUCT_URLS_FILE) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [row for row in csv.DictReader(file) if text(row.get("product_url")) != "null"]


def write_products(products: Iterable[dict[str, str]], path: Path = DISCOVERED_URLS_FILE) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["brand", "product_url", "wifi_version", "net_port"])
        writer.writeheader()
        writer.writerows(products)


def write_rows(rows: Iterable[dict[str, str]], path: Path = OUTPUT_FILE) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "null") for column in CSV_COLUMNS})


def tmall_item_id(url: str) -> str:
    return parse_qs(urlparse(url).query).get("id", ["unknown"])[0]

# 规范化天猫商品链接
def clean_tmall_url(url: str) -> str:
    item_id = tmall_item_id(url)
    return f"https://detail.tmall.com/item.htm?id={item_id}" if item_id != "unknown" else url

# 按多个选择器获取页面中的第一条有效文本
def first_text(page, selectors: list[str]) -> str:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                value = text(locator.inner_text(timeout=3000))
                if value != "null":
                    return value
        except Exception:
            pass
    return "null"


def product_title(page) -> str:
    title = first_text(page, ["h1", "[class*='ItemHeader']", "[class*='mainTitle']", "[class*='title']"])
    if title != "null":
        return title
    try:
        return text(page.title())
    except Exception:
        return "null"


def product_price(page) -> str:
    value = first_text(page, ["[class*='Price']", "[class*='price']", "[class*='priceText']"])
    if price(value) != "null":
        return value
    try:
        body = page.locator("body").inner_text(timeout=5000)
    except Exception:
        return "null"
    match = re.search(r"(?:¥|￥)\s*(\d+(?:\.\d+)?)", body)
    return match.group(1) if match else "null"

# 点击页面中指定文本对应的按钮
def click_text(page, labels: list[str], wait_ms: int = 1200) -> bool:
    for label in labels:
        try:
            target = page.get_by_text(label, exact=False).first
            if target.count() > 0 and target.is_visible(timeout=800):
                target.click(timeout=2500)
                page.wait_for_timeout(wait_ms)
                return True
        except Exception:
            pass
    return False

# 从当前页面提取评论内容
def collect_comments_from_current_view(page, comments: list[tuple[str, str]], seen: set[str], limit: int, fallback_star: str) -> int:
    added = 0
    for selector in COMMENT_SELECTORS:
        try:
            locators = page.locator(selector)
            for index in range(min(locators.count(), limit * 10)):
                comment = text(locators.nth(index).inner_text(timeout=1000))
                if not is_valid_comment(comment) or comment in seen:
                    continue
                seen.add(comment)
                comments.append((infer_star(comment, fallback_star), comment))
                added += 1
                if len(comments) >= limit:
                    return added
        except Exception:
            pass
    return added

# 提取商品评论（支持翻页和分类）
def extract_comments(page, limit: int) -> list[tuple[str, str]]:
    click_text(page, ["评价", "累计评价", "商品评价", "评论"])
    comments: list[tuple[str, str]] = []
    seen: set[str] = set()

    for group_label, fallback_star in REVIEW_GROUPS:
        click_text(page, [group_label])
        stagnant_rounds = 0
        for round_index in range(8):
            before = len(comments)
            collect_comments_from_current_view(page, comments, seen, limit, fallback_star)
            if len(comments) >= limit:
                return comments
            stagnant_rounds = stagnant_rounds + 1 if len(comments) == before else 0

            page.mouse.wheel(0, 1800)
            page.wait_for_timeout(1200)
            if round_index % 3 == 2:
                click_text(page, ["查看更多", "展开更多", "加载更多", "下一页", "下页"], wait_ms=1800)
            if stagnant_rounds >= 3:
                break
    return comments

# 根据关键词搜索并发现商品
def discover_products(keyword: str, count: int, login_wait: int) -> list[dict[str, str]]:
    from playwright.sync_api import sync_playwright

    products: list[dict[str, str]] = []
    seen: set[str] = set()
    search_url = f"https://list.tmall.com/search_product.htm?q={quote(keyword)}"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_context(locale="zh-CN").new_page()
        page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(5000)
        if login_wait > 0:
            print(f"Waiting {login_wait} seconds for manual login if needed...")
            page.wait_for_timeout(login_wait * 1000)

        for _ in range(10):
            links = page.locator("a[href*='detail.tmall.com/item.htm']")
            for index in range(min(links.count(), 120)):
                try:
                    href = text(links.nth(index).get_attribute("href", timeout=1000))
                    title = text(links.nth(index).inner_text(timeout=1000))
                except Exception:
                    continue
                if href.startswith("//"):
                    href = "https:" + href
                clean_url = clean_tmall_url(href)
                item_id = tmall_item_id(clean_url)
                if item_id == "unknown" or item_id in seen:
                    continue
                if title != "null" and not any(word in title for word in PRODUCT_HINTS):
                    continue
                seen.add(item_id)
                products.append(
                    {
                        "brand": brand(title),
                        "product_url": clean_url,
                        "wifi_version": "WiFi6",
                        "net_port": net_port(title),
                    }
                )
                if len(products) >= count:
                    browser.close()
                    write_products(products)
                    return products
            page.mouse.wheel(0, 1800)
            page.wait_for_timeout(1200)
        browser.close()

    write_products(products)
    return products

# 爬取商品详情和评论数据
def crawl_products(products: list[dict[str, str]], max_comments: int, login_wait: int) -> list[dict[str, str]]:
    from playwright.sync_api import sync_playwright

    rows: list[dict[str, str]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_context(locale="zh-CN").new_page()
        waited = False

        for product in products:
            url = text(product.get("product_url"))
            print(f"Opening item {tmall_item_id(url)}")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(5000)
                if login_wait > 0 and not waited:
                    print(f"Waiting {login_wait} seconds for manual login if needed...")
                    page.wait_for_timeout(login_wait * 1000)
                    waited = True
            except Exception as exc:
                print(f"Open failed: {url} ({exc})")
                continue

            title = product_title(page)
            body = first_text(page, ["body"])
            comments = extract_comments(page, max_comments)
            if not comments:
                comments = [("null", "null")]

            for score, comment in comments:
                rows.append(
                    make_row(
                        title,
                        brand(title, product.get("brand")),
                        product_price(page),
                        wifi_version(body, product.get("wifi_version", "WiFi6")),
                        net_port(body, product.get("net_port")),
                        score,
                        comment,
                    )
                )
            print(f"Collected {len(comments)} comments from this product.")

        browser.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Tmall router review data.")
    parser.add_argument("--real", action="store_true", help="crawl products from product_urls.csv")
    parser.add_argument("--discover", action="store_true", help="discover products from Tmall search before crawling")
    parser.add_argument("--keyword", default="WiFi6 千兆路由器")
    parser.add_argument("--product-count", type=int, default=5)
    parser.add_argument("--max-comments", type=int, default=40)
    parser.add_argument("--login-wait", type=int, default=0)
    args = parser.parse_args()

    if args.discover:
        products = discover_products(args.keyword, args.product_count, args.login_wait)
        print(f"Discovered {len(products)} products. Links saved to {DISCOVERED_URLS_FILE}")
    elif args.real:
        products = read_products()
    else:
        raise SystemExit("Please use --real or --discover. Sample mode has been removed.")

    rows = crawl_products(products, args.max_comments, args.login_wait)
    write_rows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
