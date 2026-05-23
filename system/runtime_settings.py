"""
system/runtime_settings.py

文件作用:
    运行时可热更新参数。
"""

from typing import Any, Dict

from fastapi import HTTPException

from system.config import (
    AI_ENHANCED_DEFAULT,
    AI_ENHANCE_API_KEY,
    AI_ENHANCE_TIMEOUT_SEC,
)

runtime_ai_settings: Dict[str, Any] = {
    "enabled": bool(AI_ENHANCED_DEFAULT),
    "timeout": float(AI_ENHANCE_TIMEOUT_SEC),
}


def apply_ai_runtime_settings(ai_enhanced: Any = None, timeout: Any = None) -> None:
    """应用并校验 AI 运行时参数。"""
    if ai_enhanced is not None:
        if bool(ai_enhanced) and not AI_ENHANCE_API_KEY:
            raise HTTPException(status_code=400, detail="AI 增强开启失败：未配置 ARK_API_KEY。")
        runtime_ai_settings["enabled"] = bool(ai_enhanced)
    if timeout is not None:
        val = float(timeout)
        if val < 1 or val > 120:
            raise HTTPException(status_code=400, detail="超时时间必须在 1 到 120 秒之间。")
        runtime_ai_settings["timeout"] = val
