"""Todo 插件 — 待办事项管理 + Bot 待办看板。

提供命令、Action、Service 供 bot 和用户管理待办事项。
同时将待办注入 bot 系统提示词，让 bot 实时感知自己的待办和计划。
"""

from __future__ import annotations

import asyncio
from typing import cast

from src.core.components import BasePlugin, register_plugin
from src.core.prompt import get_system_reminder_store
from src.kernel.concurrency import get_task_manager
from src.kernel.logger import get_logger

from .action import (
    AddUserTodoAction,
    ListBotTodosAction,
    ListUserTodosAction,
    MarkTodoDoneAction,
    ScheduleBotTaskAction,
)
from .command import TodoCommand
from .config import TodoPluginConfig
from .event_handler import RelayTodoEventHandler
from .service import BotTodoService, TodoService

logger = get_logger("todo_plugin")

_BOARD_BUCKET = "actor"
_BOARD_NAME = "待办事项看板"


@register_plugin
class TodoPlugin(BasePlugin):
    """Todo 插件。"""

    plugin_name: str = "todo_plugin"
    plugin_description: str = "待办事项管理 + Bot 待办看板：添加、列出、完成、提醒、自我计划"
    plugin_version: str = "1.0.0"

    configs: list[type] = [TodoPluginConfig]
    dependent_components: list[str] = []

    def __init__(self, config: TodoPluginConfig | None = None) -> None:
        super().__init__(config)
        self._schedule_id: str | None = None
        self._register_task_id: str | None = None

    def get_components(self) -> list[type]:
        return [
            TodoService,
            BotTodoService,
            TodoCommand,
            RelayTodoEventHandler,
            AddUserTodoAction,
            ListUserTodosAction,
            MarkTodoDoneAction,
            ScheduleBotTaskAction,
            ListBotTodosAction,
        ]

    async def on_plugin_loaded(self) -> None:
        # 初始同步看板
        await sync_todo_board()
        tm = get_task_manager()
        task = tm.create_task(
            self._register_schedule_when_ready(),
            name="todo_plugin_register_schedule",
            daemon=True,
        )
        self._register_task_id = task.task_id

    async def on_plugin_unloaded(self) -> None:
        from src.kernel.scheduler import get_unified_scheduler

        if self._schedule_id:
            try:
                await get_unified_scheduler().remove_schedule(self._schedule_id)
            except Exception:
                pass
            self._schedule_id = None

        if self._register_task_id:
            try:
                get_task_manager().cancel_task(self._register_task_id)
            except Exception:
                pass
            self._register_task_id = None

        get_system_reminder_store().delete(_BOARD_BUCKET, _BOARD_NAME)

    async def _register_schedule_when_ready(self) -> None:
        from src.kernel.scheduler import TriggerType, get_unified_scheduler

        scheduler = get_unified_scheduler()
        cfg = self.config if isinstance(self.config, TodoPluginConfig) else TodoPluginConfig()
        interval = int(cfg.general.remind_check_interval_seconds)

        for _attempt in range(600):
            try:
                sid = await scheduler.create_schedule(
                    callback=self._tick_job,
                    trigger_type=TriggerType.TIME,
                    trigger_config={"interval_seconds": interval},
                    is_recurring=True,
                    task_name="todo_plugin_tick",
                    force_overwrite=True,
                )
                self._schedule_id = sid
                logger.info(f"todo 定时任务已注册: {sid}")
                return
            except RuntimeError:
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"注册定时任务失败: {e}")
                await asyncio.sleep(2.0)

        logger.warning("等待 scheduler 就绪超时")

    async def _tick_job(self) -> None:
        """定时任务：用户提醒 + Bot 计划执行 + 看板刷新。"""
        from src.app.plugin_system.api.service_api import get_service

        # 用户提醒
        user_svc = get_service("todo_plugin:service:todo_service")
        if user_svc is not None:
            await cast(TodoService, user_svc).check_and_send_reminders()

        # Bot 自我计划执行
        bot_svc = get_service("todo_plugin:service:bot_todo_service")
        if bot_svc is not None:
            await cast(BotTodoService, bot_svc).check_and_execute_bot_tasks()

        # 刷新看板
        await sync_todo_board()


async def sync_todo_board() -> None:
    """Keep ordinary chat prompts free of cross-stream todo content."""

    content = (
        "待办和个人计划按聊天流隔离保存。"
        "普通聊天中不要臆测或复述待办内容；只有用户明确询问或调用 /待办 时才查询。"
    )

    store = get_system_reminder_store()
    store.set(_BOARD_BUCKET, _BOARD_NAME, content)
    logger.debug("待办看板已同步")
