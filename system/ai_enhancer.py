"""
system/ai_enhancer.py

文件作用:
    AI 增强回复模块。
    将上下文输入和 topk 候选答句整合后调用大模型生成最终回复。
"""

from typing import Any, Dict, List, Tuple

import requests


def _trim_text(text: str, max_len: int) -> str:
    """截断文本以降低 token 消耗并保持请求稳定。"""
    val = str(text or "").strip()
    if len(val) <= max_len:
        return val
    return val[:max_len]


def _extract_ai_output_text(resp_json: Dict[str, Any]) -> str:
    """从 responses 接口返回中提取文本。"""
    output_text = resp_json.get("output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = resp_json.get("output", [])
    if isinstance(output, list):
        for item in output:
            content = item.get("content", []) if isinstance(item, dict) else []
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    txt = part.get("text", "")
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
    return ""


def generate_ai_enhanced_reply(
    contextual_user_input: str,
    candidates: List[Tuple[float, Dict[str, Any]]],
    top_k: int,
    model_name: str,
    responses_url: str,
    api_key: str,
    timeout_sec: float,
) -> str:
    """调用 AI 生成融合后的最佳回复。"""
    # 无候选时直接返回空串，由上游统一处理兜底话术。
    if not candidates:
        return ""

    # 未配置 API Key 时回退 top1，保证线上链路可用。
    if not api_key:
        return str(candidates[0][1].get("reply", ""))

    # 为加速与降 token，AI 融合阶段只保留更小候选集。
    top_k_safe = max(1, int(top_k))
    ai_top_k = min(top_k_safe, 3)
    top_replies: List[str] = []
    for _, item in candidates[:ai_top_k]:
        top_replies.append(_trim_text(str(item.get("reply", "") or ""), max_len=72))

    top_reply_lines = [f"{idx}. {reply}" for idx, reply in enumerate(top_replies, start=1)]

    compact_query = _trim_text(str(contextual_user_input), max_len=48)
    prompt = (
        "你是中文助手。请在候选中选最合适答复并润色成一句自然中文。"
        "\n如果你觉得这些候选回答都不好，请按你的想法直接生成更好的回答。"
        "\n只输出最终答复，不要解释。"
        f"\n用户输入:{compact_query}"
        f"\n候选:\n" + "\n".join(top_reply_lines)
    )

    payload = {
        "model": model_name,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        # 调用外部模型服务；失败时回退 top1，避免因网络波动中断会话。
        resp = requests.post(responses_url, headers=headers, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        text = _extract_ai_output_text(resp.json())
        if text:
            return text
    except Exception:
        pass

    return str(candidates[0][1].get("reply", ""))
