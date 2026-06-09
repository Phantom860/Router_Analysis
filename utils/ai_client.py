import requests
import os
from dotenv import load_dotenv

load_dotenv()

AK = os.getenv("VOLC_AK")
SK = os.getenv("VOLC_SK")
MODEL_ID = os.getenv("MODEL_ID")
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

def call_doubao(prompt_text, system_prompt=None, temperature=0.3):
    """
    调用火山方舟AI接口
    
    Args:
        prompt_text: 用户提示词
        system_prompt: 系统提示词（可选）
        temperature: 温度参数（0-1）
    
    Returns:
        str: AI返回的内容
    """
    headers = {
        "Authorization": f"Bearer {AK}/{SK}",
        "Content-Type": "application/json"
    }
    
    system = system_prompt or "你是电信内部WiFi路由器采购分析助手，根据路由器参数、电商评价做选购分析、对比总结、输出导购建议，回答专业简洁，可输出表格对比优缺点。"
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": temperature,
        "max_tokens": 2000
    }
    
    try:
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        res_json = resp.json()
        
        if res_json.get("choices"):
            return res_json["choices"][0]["message"]["content"]
        else:
            return f"模型调用失败：{res_json.get('error', {}).get('message', '未知错误')}"
            
    except requests.exceptions.Timeout:
        return "接口超时，请稍后重试"
    except requests.exceptions.RequestException as e:
        return f"网络请求异常：{str(e)}"
    except Exception as e:
        return f"接口异常：{str(e)}"