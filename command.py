"""Todo 命令组件 — 提供 /todo（/待办）命令。

子命令：
    add <内容>               — 添加待办
    list [status]            — 列出待办（status: pending/done/all，默认 pending）
    done <uid>               — 标记完成（直接删除）
    undo <uid>               — 恢复为未完成
    delete <uid>             — 删除
    remind <uid> <时间>      — 设置提醒（30m / 18:00 / "2026-06-01 09:00"）
    clear                    — 清空已完成和已取消
    help                     — 帮助
"""

from __future__ import annotations

import datetime
import re
import shlex
from typing import cast

from src.app.plugin_system.api import adapter_api, send_api
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.service_api import get_service
from src.app.plugin_system.base import BaseCommand, cmd_route
from src.app.plugin_system.types import PermissionLevel
from src.core.managers.plugin_manager import get_plugin_manager

from .service import BotTodoService, TodoService, _now

logger = get_logger("todo_plugin.command")

_HELP = """\
/待办 用法：
  /待办 add <内容>               — 添加待办
  /待办 list [status]            — 列出待办提醒（pending/done/all，默认 pending）
  /待办 plans                    — 列出你自己的日程计划
  /待办 done <uid>               — 完成
  /待办 undo <uid>               — 恢复为未完成
  /待办 delete <uid>             — 删除
  /待办 remind <uid> <时间>      — 设置提醒（如 30m / 18:00 / "2026-06-01 09:00"）
  /待办 clear                    — 清空已完成和已取消
  /待办 help                     — 帮助"""

_RE_INTERVAL = re.compile(r"^(\d+)\s*(m|min|分钟?|h|小时?|d|天|s|秒)?$", re.IGNORECASE)


class TodoCommand(BaseCommand):
    """待办事项命令。"""

    command_name: str = "todo"
    command_description: str = "管理待办事项：添加、列出、完成、提醒"
    permission_level: PermissionLevel = PermissionLevel.USER

    @classmethod
    def match(cls, parts: list[str]) -> int:
        if not parts:
            return 0
        if parts[0] in ("todo", "待办"):
            return 1
        return 0

    async def execute(self, message_text: str) -> tuple[bool, str]:
        """Execute the todo command through the base router."""

        return await super().execute(message_text)

    async def _route_and_execute(self, command_text: str) -> tuple[bool, str]:
        """覆写父类方法：区分空输入（显示列表）和未匹配命令（报错提示）。

        父类 BaseCommand 的 bug：@cmd_route() 把根处理器注册在 root 节点上，
        导致所有未匹配的子命令都会 fallback 到根处理器。这里通过在遍历完成后
        检查 consumed 来区分两种情况。
        """
        try:
            parts = shlex.split(command_text)
        except ValueError as e:
            return False, f"参数解析错误: {e}"

        if not parts:
            if self._root.handler is not None:
                return await self._call_handler(self._root.handler, [])
            return False, "空命令"

        current = self._root
        consumed = 0

        for part in parts:
            if part in current.children:
                current = current.children[part]
                consumed += 1
            else:
                break

        # 修复：consumed == 0 且 parts 非空 → 输入了但没匹配到任何子命令
        if consumed == 0:
            return False, (
                f"未知命令: /{self.command_name} {parts[0]}\n"
                f"可用子命令: add, list, plans, done, undo, delete, remind, clear, help\n"
                f"输入 /待办 help 查看详细帮助"
            )

        if current.handler is None:
            return await self._generate_help(current, parts[consumed:])

        args = parts[consumed:]
        if consumed == 1 and parts[0] == "add" and args:
            args = [" ".join(args)]
        try:
            return await self._call_handler(current.handler, args)
        except Exception as e:
            return False, f"执行错误: {e}"

    # ── 工具方法 ────────────────────────────────────────────────────────────

    async def _reply(self, text: str) -> None:
        await send_api.send_text(text, stream_id=self.stream_id)

    async def _svc(self) -> TodoService:
        svc = get_service("todo_plugin:service:todo_service")
        if svc is None:
            raise RuntimeError("TodoService 未加载")
        return cast(TodoService, svc)

    async def _bot_svc(self) -> BotTodoService:
        svc = get_service("todo_plugin:service:bot_todo_service")
        if svc is None:
            raise RuntimeError("BotTodoService 未加载")
        return cast(BotTodoService, svc)

    async def _bot_plan_stream_ids(self) -> list[str]:
        """Return local bot-plan stream partitions visible to this command."""

        stream_ids = [self.stream_id]
        relay_bot_id = self._relay_config_bot_id()
        if relay_bot_id:
            stream_ids.append(f"bot_relay:{relay_bot_id}")

        platform = getattr(self._message, "platform", "") if self._message is not None else ""
        if platform:
            try:
                bot_info = await adapter_api.get_bot_info_by_platform(platform)
            except Exception:
                bot_info = None
            if bot_info and bot_info.get("bot_id"):
                stream_ids.append(f"bot_relay:{bot_info['bot_id']}")

        unique: list[str] = []
        seen: set[str] = set()
        for stream_id in stream_ids:
            if stream_id and stream_id not in seen:
                unique.append(stream_id)
                seen.add(stream_id)
        return unique

    @staticmethod
    def _relay_config_bot_id() -> str:
        """Read the loaded bot_private_relay local bot id when available."""

        relay_plugin = get_plugin_manager().get_plugin("bot_private_relay")
        relay_config = getattr(relay_plugin, "config", None)
        relay_section = getattr(relay_config, "relay", None)
        bot_id = getattr(relay_section, "bot_id", "")
        return str(bot_id or "")

    @staticmethod
    def _parse_time(raw: str) -> float | None:
        raw = raw.strip().strip('"').strip("'")
        if not raw:
            return None

        m = _RE_INTERVAL.match(raw)
        if m:
            value = int(m.group(1))
            unit = (m.group(2) or "m").lower()
            multipliers: dict[str, float] = {
                "s": 1, "秒": 1,
                "m": 60, "min": 60, "分钟": 60,
                "h": 3600, "小时": 3600,
                "d": 86400, "天": 86400,
            }
            return _now() + value * multipliers.get(unit, 60)

        formats = [
            "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M", "%m-%d %H:%M", "%m/%d %H:%M",
        ]
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(raw, fmt)
                now = datetime.datetime.now()
                if "%Y" not in fmt:
                    dt = dt.replace(year=now.year)
                if dt < now and "%Y" not in fmt:
                    dt = dt.replace(year=now.year + 1)
                return dt.timestamp()
            except ValueError:
                continue

        try:
            parts_ = raw.split(":")
            if len(parts_) == 2:
                hour, minute = int(parts_[0]), int(parts_[1])
                now = datetime.datetime.now()
                dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if dt <= now:
                    dt += datetime.timedelta(days=1)
                return dt.timestamp()
        except (ValueError, OverflowError):
            pass

        return None

    @staticmethod
    def _format_time(ts: float | None) -> str:
        if ts is None:
            return "无提醒"
        return datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")

    @staticmethod
    def _todo_time_label(todo: dict[str, object]) -> str:
        """Format the visible time for a todo item."""

        remind_at = todo.get("remind_at")
        if isinstance(remind_at, (int, float)):
            return f"提醒时间：{TodoCommand._format_time(float(remind_at))}"

        created_at = todo.get("created_at")
        if isinstance(created_at, (int, float)):
            return f"记录时间：{TodoCommand._format_time(float(created_at))}"

        return "记录时间：未知"

    @staticmethod
    def _status_label(status: str) -> str:
        return {"pending": "[ ]", "done": "[✓]", "cancelled": "[x]"}.get(status, "[?]")

    # ── 子命令路由 ──────────────────────────────────────────────────────────

    @cmd_route()
    async def handle_root(self) -> tuple[bool, str]:
        svc = await self._svc()
        todos = await svc.list_todos(self.stream_id, "pending")
        if not todos:
            await self._reply("暂无待办事项。\n输入 /待办 help 查看用法")
            return True, "ok"

        lines = ["待办列表："]
        for t in todos:
            uid = t.get("todo_uid", "")
            content = str(t.get("content", ""))[:40]
            lines.append(
                f"  {self._status_label(t.get('status', 'pending'))} "
                f"{uid} {content}（{self._todo_time_label(t)}）"
            )
        await self._reply("\n".join(lines))
        return True, "ok"

    @cmd_route("add")
    async def handle_add(self, content: str = "") -> tuple[bool, str]:
        if not content.strip():
            await self._reply("用法：/待办 add <内容>")
            return False, "missing content"

        svc = await self._svc()
        item = await svc.add_todo(stream_id=self.stream_id, content=content)
        if item is None:
            await self._reply("待办已达上限，请先清理已完成事项")
            return False, "max_items"
        await self._reply(f"已添加待办 [{item['todo_uid']}] {content}")
        return True, "ok"

    @cmd_route("list")
    async def handle_list(self, status: str = "") -> tuple[bool, str]:
        status = status.strip().lower()
        if status and status not in ("pending", "done", "all"):
            await self._reply("状态只能是 pending / done / all")
            return False, "invalid status"

        svc = await self._svc()
        filter_status = "" if status == "all" else (status or "pending")
        todos = await svc.list_todos(self.stream_id, filter_status)
        if not todos:
            label = {"pending": "待处理", "done": "已完成", "": "全部"}.get(filter_status, "匹配")
            await self._reply(f"暂无{label}待办")
            return True, "ok"

        lines = ["待办列表："]
        for t in todos:
            uid = t.get("todo_uid", "")
            content = str(t.get("content", ""))[:40]
            p = t.get("priority", 3)
            stars = "★" * p + "☆" * (5 - p)
            lines.append(
                f"  {self._status_label(t.get('status', 'pending'))}"
                f" [{stars}] {uid} {content}（{self._todo_time_label(t)}）"
            )
        await self._reply("\n".join(lines))
        return True, "ok"

    @cmd_route("plans")
    async def handle_plans(self) -> tuple[bool, str]:
        """列出你自己的日程计划。"""
        svc = await self._bot_svc()
        plans = await svc.list_bot_todos_for_streams(await self._bot_plan_stream_ids(), "pending")
        if not plans:
            await self._reply("暂无个人日程计划")
            return True, "ok"

        lines = ["你的日程计划："]
        for p in plans:
            uid = p.get("bot_todo_uid", "")
            plan = str(p.get("plan", ""))
            ts = p.get("scheduled_at", 0)
            dt_str = self._format_time(ts) if ts else "未知"
            lines.append(f"  ○ [{uid}] {plan}（{dt_str}）")
        await self._reply("\n".join(lines))
        return True, "ok"

    @cmd_route("done")
    async def handle_done(self, todo_uid: str = "") -> tuple[bool, str]:
        if not todo_uid.strip():
            await self._reply("用法：/待办 done <uid>")
            return False, "missing uid"

        svc = await self._svc()
        ok = await svc.mark_done(self.stream_id, todo_uid.strip())
        if ok:
            await self._reply(f"已完成: {todo_uid}")
            return True, "ok"
        await self._reply(f"未找到待办: {todo_uid}")
        return False, "not found"

    @cmd_route("undo")
    async def handle_undo(self, todo_uid: str = "") -> tuple[bool, str]:
        if not todo_uid.strip():
            await self._reply("用法：/待办 undo <uid>")
            return False, "missing uid"

        svc = await self._svc()
        ok = await svc.mark_undo(self.stream_id, todo_uid.strip())
        if ok:
            await self._reply(f"已恢复: {todo_uid}")
            return True, "ok"
        await self._reply(f"未找到待办: {todo_uid}")
        return False, "not found"

    @cmd_route("delete")
    async def handle_delete(self, todo_uid: str = "") -> tuple[bool, str]:
        if not todo_uid.strip():
            await self._reply("用法：/待办 delete <uid>")
            return False, "missing uid"

        svc = await self._svc()
        ok = await svc.delete_todo(self.stream_id, todo_uid.strip())
        if ok:
            await self._reply(f"已删除: {todo_uid}")
            return True, "ok"
        await self._reply(f"未找到待办: {todo_uid}")
        return False, "not found"

    @cmd_route("remind")
    async def handle_remind(self, todo_uid: str = "", time_str: str = "") -> tuple[bool, str]:
        if not todo_uid.strip() or not time_str.strip():
            await self._reply(
                "用法：/待办 remind <uid> <时间>\n"
                "时间格式：30m / 5min / 3h / 2d / 18:00 / \"2026-06-01 09:00\""
            )
            return False, "missing args"

        remind_at = self._parse_time(time_str)
        if remind_at is None:
            await self._reply(f"无法解析时间: {time_str}\n支持格式: 30m / 3h / 2d / 18:00 / \"2026-06-01 09:00\"")
            return False, "invalid time"

        svc = await self._svc()
        ok = await svc.set_reminder(self.stream_id, todo_uid.strip(), remind_at)
        if ok:
            await self._reply(f"已设置提醒 [{todo_uid}] -> {self._format_time(remind_at)}")
            return True, "ok"
        await self._reply(f"未找到待办: {todo_uid}")
        return False, "not found"

    @cmd_route("clear")
    async def handle_clear(self) -> tuple[bool, str]:
        svc = await self._svc()
        count = await svc.clear_completed(self.stream_id)
        if count > 0:
            await self._reply(f"已清理 {count} 条已完成/已取消的待办")
        else:
            await self._reply("没有需要清理的事项")
        return True, "ok"

    @cmd_route("help")
    async def handle_help(self) -> tuple[bool, str]:
        await self._reply(_HELP)
        return True, "ok"
