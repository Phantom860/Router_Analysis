import requests
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 方舟固定接口地址
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
# 读取密钥与模型
API_KEY = os.getenv("VOLC_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

def call_doubao(prompt_text, system_prompt=None, temperature=0.3, max_tokens=2000):
    """
    火山方舟豆包对话接口标准调用
    :param prompt_text: 用户提问内容
    :param system_prompt: 角色系统提示词
    :param temperature: 随机性 0~1，越低越严谨
    :param max_tokens: 最大输出长度
    :return: AI回答文本/错误信息
    """
    # 基础校验
    if not API_KEY or not MODEL_ID:
        return "错误：请检查.env配置 VOLC_API_KEY 和 MODEL_ID"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 默认系统提示
    default_system = "你是电信内部WiFi路由器采购分析助手，根据路由器参数、电商评价做选购分析、对比总结、输出导购建议，回答专业简洁，可输出表格对比优缺点。"
    system_content = system_prompt if system_prompt else default_system

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt_text}
    ]

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        resp.encoding = "utf-8"
        resp_json = resp.json()

        # 成功返回内容
        if resp.status_code == 200:
            return resp_json["choices"][0]["message"]["content"]
        else:
            # 打印方舟返回的详细错误信息，方便排错
            return f"请求失败[{resp.status_code}]：{resp_json.get('error', resp.text)}"

    except requests.exceptions.Timeout:
        return "请求超时（60s），可适当加大timeout时长"
    except requests.exceptions.ConnectionError:
        return "网络连接失败，请检查网络或接口地址"
    except Exception as e:
        return f"程序异常：{str(e)}"
