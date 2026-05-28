"""Todo 插件配置。"""

from __future__ import annotations

from typing import ClassVar

from src.core.components.base.config import BaseConfig, Field, SectionBase, config_section


class TodoPluginConfig(BaseConfig):
    """Todo 插件配置。

    配置文件路径：config/plugins/todo_plugin/config.toml
    """

    config_name: ClassVar[str] = "config"
    config_description: ClassVar[str] = "待办事项插件配置"

    @config_section("general")
    class GeneralSection(SectionBase):
        """通用配置。"""

        max_items_per_user: int = Field(
            default=100,
            description="每个用户最大待办事项数量",
        )
        remind_check_interval_seconds: int = Field(
            default=60,
            description="提醒检查间隔（秒）",
        )

    @config_section("bot_execution")
    class BotExecutionSection(SectionBase):
        """Bot 计划执行配置。"""

        relay_retry_enabled: bool = Field(
            default=True,
            description="relay 来源 Bot 计划执行失败后是否重试",
        )
        relay_max_retries: int = Field(
            default=2,
            description="relay 来源 Bot 计划执行失败后的最大重试次数",
        )
        relay_retry_delay_seconds: int = Field(
            default=60,
            description="relay 来源 Bot 计划执行失败后的重试间隔秒数",
        )

    general: GeneralSection = Field(default_factory=GeneralSection)
    bot_execution: BotExecutionSection = Field(default_factory=BotExecutionSection)
