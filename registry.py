"""Bot 执行工具注册表。

其他插件可通过 register_bot_tool() 向 bot 自我执行 LLM 注入工具。
"""

from __future__ import annotations

_bot_tools: list[type] = []


def register_bot_tool(tool_cls: type) -> None:
    """注册一个工具到 bot 自我执行 LLM。

    其他插件在 on_plugin_loaded 中调用此函数即可注入工具。
    重复注册同一个类会被忽略。

    Example:
        >>> from plugins.todo_plugin.registry import register_bot_tool
        >>> register_bot_tool(MyCustomTool)
    """
    if tool_cls not in _bot_tools:
        _bot_tools.append(tool_cls)


def get_bot_tools() -> list[type]:
    """获取所有已注册的 bot 执行工具。"""
    return list(_bot_tools)
