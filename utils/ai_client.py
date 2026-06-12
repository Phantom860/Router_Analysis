"""
ai_client.py —— 唯一职责：封装火山方舟豆包 API 调用

对外只暴露一个函数 call_doubao()，业务逻辑、prompt 拼装全在调用方。
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

_API_URL  = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
_API_KEY  = os.getenv("VOLC_API_KEY")
_MODEL_ID = os.getenv("MODEL_ID")

# 默认系统提示，调用方可通过 system_prompt 参数覆盖
_DEFAULT_SYSTEM = (
    "你是电信内部WiFi路由器采购分析助手，"
    "根据路由器参数、电商评价做选购分析、对比总结、输出导购建议，"
    "回答专业简洁，可输出表格对比优缺点。"
)


def call_doubao(
    prompt_text: str,
    system_prompt: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """
    调用豆包大模型，返回回答文本。
    失败时返回可读错误字符串（不抛异常，让页面能优雅降级）。

    Args:
        prompt_text:   用户侧 prompt
        system_prompt: 系统提示，传 None 时使用默认角色
        temperature:   0~1，越低越严谨
        max_tokens:    最大输出 token 数

    Returns:
        str: AI 回答 / 错误描述
    """
    if not _API_KEY or not _MODEL_ID:
        return "错误：请检查 .env 中的 VOLC_API_KEY 和 MODEL_ID"

    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _MODEL_ID,
        "messages": [
            {"role": "system",  "content": system_prompt or _DEFAULT_SYSTEM},
            {"role": "user",    "content": prompt_text},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    try:
        resp = requests.post(_API_URL, json=payload, headers=headers, timeout=60)
        resp.encoding = "utf-8"
        resp_json = resp.json()

        if resp.status_code == 200:
            return resp_json["choices"][0]["message"]["content"]
        return f"请求失败[{resp.status_code}]：{resp_json.get('error', resp.text)}"

    except requests.exceptions.Timeout:
        return "请求超时（60s），可适当加大 timeout 时长"
    except requests.exceptions.ConnectionError:
        return "网络连接失败，请检查网络或接口地址"
    except Exception as e:
        return f"程序异常：{e}"