"""
system/runtime_settings.py

文件作用:
    运行时可热更新参数。
    目前聚焦 AI 增强开关。
"""

from typing import Any, Dict

from fastapi import HTTPException

from system.config import (
    AI_ENHANCED_DEFAULT,
    AI_ENHANCE_MODEL_NAME,
    AI_ENHANCE_RESPONSES_URL,
    AI_ENHANCE_API_KEY,
)

runtime_ai_settings: Dict[str, Any] = {
    "enabled": bool(AI_ENHANCED_DEFAULT),
}


def build_ai_meta_payload() -> Dict[str, Any]:
    """返回 AI 运行时状态与元信息。"""
    return {
        "ai_enhanced": bool(runtime_ai_settings.get("enabled", False)),
        "ai_model_name": str(AI_ENHANCE_MODEL_NAME),
        "ai_responses_url": str(AI_ENHANCE_RESPONSES_URL),
        "ai_api_ready": bool(AI_ENHANCE_API_KEY),
    }


def apply_ai_runtime_settings(ai_enhanced: Any = None) -> None:
    """应用并校验 AI 运行时参数。"""
    # 开关变更：开启前校验 API Key 可用性，避免“开启成功但调用必失败”。
    if ai_enhanced is not None:
        if bool(ai_enhanced) and not AI_ENHANCE_API_KEY:
            raise HTTPException(status_code=400, detail="AI 增强开启失败：未配置 ARK_API_KEY。")
        runtime_ai_settings["enabled"] = bool(ai_enhanced)

