"""Todo 服务 — 待办事项的核心 CRUD 与 LLM 提醒，以及 Bot 自我待办。

基于 storage_api 的 JSON 持久化，按 stream_id 分区存储。
- TodoService: 用户待办的增删改查与 LLM 拟人化提醒
- BotTodoService: Bot 自己的计划任务，到期通过 LLM 生成行为
"""

from __future__ import annotations

import asyncio
import datetime
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from src.app.plugin_system.api import send_api, storage_api
from src.app.plugin_system.api.llm_api import create_llm_request, exec_llm_usable, get_model_set_by_task
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseService
from src.core.config import get_core_config
from src.kernel.llm import LLMPayload, ROLE, Text, ToolResult

from .config import TodoPluginConfig

logger = get_logger("todo_plugin.service")

_STORE_NAME = "todo_plugin"
_DATA_KEY = "todos"
_BOT_DATA_KEY = "bot_todos"
_lock = asyncio.Lock()
_bot_lock = asyncio.Lock()


@dataclass(slots=True)
class BotPlanExecutionResult:
    """Structured result from one bot plan execution attempt."""

    ok: bool
    text: str
    required_tool_called: bool = False
    required_tool_succeeded: bool = False
    error: str = ""


def _uid() -> str:
    """Return a short opaque todo identifier."""

    return uuid.uuid4().hex[:8]


async def _load_all() -> dict[str, list[dict[str, Any]]]:
    """Load all user todo partitions from plugin storage."""

    data = await storage_api.load_json(_STORE_NAME, _DATA_KEY)
    return data if data is not None else {}


async def _save_all(data: dict[str, list[dict[str, Any]]]) -> None:
    """Persist all user todo partitions to plugin storage."""

    await storage_api.save_json(_STORE_NAME, _DATA_KEY, data)


def _now() -> float:
    """Return the current Unix timestamp."""

    return time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# Recurrence 解析器
# ═══════════════════════════════════════════════════════════════════════════════

_RECURRENCE_TYPE = ("daily", "weekly", "interval", "cron")
_WEEKDAY_NAMES: dict[str, int] = {
    "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
}
_DEFAULT_DAILY_TIME = "08:00"
_DEFAULT_WEEKLY_TIME = "09:00"

_RE_DAILY = re.compile(r"每[日天]")
_RE_WEEKLY = re.compile(r"每周")
_RE_WEEKDAY = re.compile(r"每个工作日")
_RE_WEEKEND = re.compile(r"每个周末")
_RE_INTERVAL_NUM = re.compile(r"每\s*(\d+)\s*(分钟?|min|小时?|h|天|d|秒|s)")
_RE_HOUR_MIN = re.compile(r"(\d{1,2})[点:：](\d{2})?")
_RE_FUZZY_TIME = re.compile(r"(早上|上午|中午|下午|晚上|傍晚|黎明|凌晨)")


def _parse_recurrence(text: str) -> dict[str, Any] | None:
    """从 LLM 传入的 repeat 参数或 content 兜底提取循环规则。

    返回值字段：
        recurrence_type: str
        recurrence_time: str | None   "HH:MM"
        recurrence_weekdays: list[int] | None
        recurrence_interval: int | None   秒
        recurrence_cron: str | None
        recurrence_max_count: int | None
        recurrence_end_at: float | None
    """

    text = (text or "").strip()
    if not text:
        return None

    result: dict[str, Any] = {
        "recurrence_type": None,
        "recurrence_time": None,
        "recurrence_weekdays": None,
        "recurrence_interval": None,
        "recurrence_cron": None,
        "recurrence_max_count": None,
        "recurrence_end_at": None,
    }

    # 检测基本类型
    is_weekly = bool(_RE_WEEKLY.search(text))
    is_weekday = bool(_RE_WEEKDAY.search(text))
    is_weekend = bool(_RE_WEEKEND.search(text))
    is_daily = bool(_RE_DAILY.search(text))
    interval_m = _RE_INTERVAL_NUM.search(text)
    is_interval = bool(interval_m)

    if not any([is_daily, is_weekly, is_weekday, is_weekend, is_interval]):
        return None

    # ── 提取时间点 ──
    hour_min_m = _RE_HOUR_MIN.search(text)
    if hour_min_m:
        h = int(hour_min_m.group(1))
        m = int(hour_min_m.group(2)) if hour_min_m.group(2) else 0
        if h < 24 and m < 60:
            result["recurrence_time"] = f"{h:02d}:{m:02d}"
    else:
        fuzzy_m = _RE_FUZZY_TIME.search(text)
        if fuzzy_m:
            fuzzy = fuzzy_m.group(1)
            fuzzy_map = {
                "早上": "08:00", "上午": "09:00", "中午": "12:00",
                "下午": "14:00", "晚上": "20:00", "傍晚": "18:00",
                "黎明": "06:00", "凌晨": "02:00",
            }
            result["recurrence_time"] = fuzzy_map.get(fuzzy, _DEFAULT_DAILY_TIME)

    # ── 提取周日 ──
    if is_weekly:
        weekday_chars = re.findall(r"[一二三四五六日天]", text)
        result["recurrence_weekdays"] = [_WEEKDAY_NAMES[c] for c in weekday_chars] if weekday_chars else None

    # ── 提取 max_count ──
    max_count_m = re.search(r"最多\s*(\d+)\s*次", text)
    if max_count_m:
        result["recurrence_max_count"] = int(max_count_m.group(1))

    # ── 提取 end_at ──
    end_m = re.search(r"到\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?\s*(止|截至|为止|结束)?", text)
    if end_m:
        month = int(end_m.group(1))
        day = int(end_m.group(2))
        now = datetime.datetime.now()
        end = now.replace(month=month, day=day, hour=23, minute=59, second=59)
        if end < now:
            end = end.replace(year=now.year + 1)
        result["recurrence_end_at"] = end.timestamp()

    # ── 确定类型 ──
    if is_weekday:
        result["recurrence_type"] = "weekly"
        result["recurrence_weekdays"] = [0, 1, 2, 3, 4]
    elif is_weekend:
        result["recurrence_type"] = "weekly"
        result["recurrence_weekdays"] = [5, 6]
    elif is_weekly:
        result["recurrence_type"] = "weekly"
    elif is_interval:
        result["recurrence_type"] = "interval"
        val = int(interval_m.group(1))
        unit = (interval_m.group(2) or "分钟").strip()
        multipliers: dict[str, int] = {
            "秒": 1, "s": 1,
            "分钟": 60, "分": 60, "min": 60,
            "小时": 3600, "时": 3600, "h": 3600,
            "天": 86400, "d": 86400,
        }
        result["recurrence_interval"] = val * multipliers.get(unit, 60)
    elif is_daily:
        result["recurrence_type"] = "daily"
        if result["recurrence_time"] is None:
            result["recurrence_time"] = _DEFAULT_DAILY_TIME
    else:
        return None

    return result


def _next_recurrence_time(item: dict[str, Any]) -> float:
    """根据已执行项的 recurrence 字段计算下一次触发时间戳。

    调用前提：item 的提醒/执行刚完成，remind_at / scheduled_at 是上一次的时间戳。
    """

    rtype = item.get("recurrence_type")
    if not rtype or rtype not in _RECURRENCE_TYPE:
        return _now()

    last_ts = float(item.get("remind_at") or item.get("scheduled_at") or _now())
    last_dt = datetime.datetime.fromtimestamp(last_ts)
    time_str = str(item.get("recurrence_time") or "")
    hour = last_dt.hour
    minute = last_dt.minute
    if time_str:
        try:
            parts = time_str.split(":")
            hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            pass

    if rtype == "daily":
        next_dt = last_dt + datetime.timedelta(days=1)
        next_dt = next_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return next_dt.timestamp()

    if rtype == "weekly":
        weekdays = item.get("recurrence_weekdays")
        if isinstance(weekdays, list) and len(weekdays) > 0:
            for offset in range(1, 8):
                candidate = last_dt + datetime.timedelta(days=offset)
                if candidate.weekday() in weekdays:
                    candidate = candidate.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    return candidate.timestamp()
            # 没找到匹配 → 回退 7 天
            next_dt = last_dt + datetime.timedelta(days=7)
        else:
            next_dt = last_dt + datetime.timedelta(days=7)
        next_dt = next_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return next_dt.timestamp()

    if rtype == "interval":
        interval = int(item.get("recurrence_interval") or 3600)
        return last_ts + interval

    if rtype == "cron":
        cron_expr = str(item.get("recurrence_cron") or "")
        if not cron_expr:
            return _now()
        try:
            from croniter import croniter

            return croniter(cron_expr, last_dt).get_next(float)
        except ImportError:
            logger.warning("croniter 未安装，cron 循环任务无法计算下次时间，退回到立即执行")
            return _now()
        except Exception as exc:
            logger.warning(f"croniter 计算失败: {exc}")
            return _now()

    return _now()


def _build_recurrence_fields(recurrence: dict[str, Any] | None) -> dict[str, Any]:
    """标准化 recurrence 字段为存储用的 dict。"""

    if recurrence is None:
        return {
            "recurrence_type": None,
            "recurrence_time": None,
            "recurrence_weekdays": None,
            "recurrence_interval": None,
            "recurrence_cron": None,
            "recurrence_max_count": None,
            "recurrence_executed_count": 0,
            "recurrence_end_at": None,
        }

    return {
        "recurrence_type": recurrence.get("recurrence_type"),
        "recurrence_time": recurrence.get("recurrence_time"),
        "recurrence_weekdays": recurrence.get("recurrence_weekdays"),
        "recurrence_interval": recurrence.get("recurrence_interval"),
        "recurrence_cron": recurrence.get("recurrence_cron"),
        "recurrence_max_count": recurrence.get("recurrence_max_count"),
        "recurrence_executed_count": 0,
        "recurrence_end_at": recurrence.get("recurrence_end_at"),
    }


def _recurrence_done(item: dict[str, Any]) -> bool:
    """检查循环任务是否达到上限或截止日期，应永久完成。"""

    rtype = item.get("recurrence_type")
    if not rtype:
        return False

    max_count = item.get("recurrence_max_count")
    executed = int(item.get("recurrence_executed_count") or 0)
    if isinstance(max_count, int) and max_count > 0 and executed >= max_count:
        return True

    end_at = item.get("recurrence_end_at")
    now_ts = _now()
    if isinstance(end_at, (int, float)) and end_at < now_ts:
        return True

    return False


class TodoService(BaseService):
    """待办事项服务。

    身份键统一使用 stream_id，天然隔离不同平台/用户。
    其他插件通过 get_service("todo_plugin:service:todo_service") 调用。
    """

    service_name: str = "todo_service"
    service_description: str = "待办事项的增删改查与 LLM 拟人化提醒服务"
    version: str = "1.0.0"

    def _max_items(self) -> int:
        cfg = self.plugin.config
        if isinstance(cfg, TodoPluginConfig):
            return int(cfg.general.max_items_per_user)
        return 100

    # ── 公开 API ────────────────────────────────────────────────────────────

    async def add_todo(
        self,
        *,
        stream_id: str,
        content: str,
        priority: int = 3,
        remind_at: float | None = None,
        extra: dict[str, Any] | None = None,
        recurrence: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """添加一条待办事项。

        Args:
            stream_id: 聊天流 ID（身份 + 路由）
            content: 待办内容
            priority: 优先级 1-5
            remind_at: 提醒时间戳，可选
            extra: 附加字段，供插件桥接记录来源信息
            recurrence: 循环规则，由 _parse_recurrence 解析得到

        Returns:
            创建的待办条目；超出上限返回 None
        """
        content = content.strip()
        if not content:
            raise ValueError("待办内容不能为空")
        priority = max(1, min(5, priority))

        async with _lock:
            data = await _load_all()
            user_todos: list[dict[str, Any]] = data.get(stream_id, [])

            if len(user_todos) >= self._max_items():
                return None

            item: dict[str, Any] = {
                "todo_uid": _uid(),
                "content": content,
                "status": "pending",
                "priority": priority,
                "remind_at": remind_at,
                "reminded": False,
                "created_at": _now(),
            }
            item.update(_build_recurrence_fields(recurrence))
            if extra:
                item.update(extra)
            user_todos.append(item)
            data[stream_id] = user_todos
            await _save_all(data)
            logger.debug(f"添加待办: {stream_id} -> {item['todo_uid']}")
            return item

    async def list_todos(
        self,
        stream_id: str,
        status_filter: str = "",
    ) -> list[dict[str, Any]]:
        """列出待办事项。"""
        data = await _load_all()
        todos = data.get(stream_id, [])
        if status_filter and status_filter != "all":
            todos = [t for t in todos if t.get("status") == status_filter]
        return sorted(todos, key=lambda t: (-t.get("priority", 3), t.get("created_at", 0)))

    async def mark_done(self, stream_id: str, todo_uid: str) -> bool:
        """Mark a user todo as done."""

        return await self._update_status(stream_id, todo_uid, "done")

    async def mark_undo(self, stream_id: str, todo_uid: str) -> bool:
        """Restore a user todo to pending."""

        return await self._update_status(stream_id, todo_uid, "pending")

    async def cancel_todo(self, stream_id: str, todo_uid: str) -> bool:
        """Mark a user todo as cancelled."""

        return await self._update_status(stream_id, todo_uid, "cancelled")

    async def delete_todo(self, stream_id: str, todo_uid: str) -> bool:
        """Delete a user todo by id."""

        async with _lock:
            data = await _load_all()
            todos = data.get(stream_id, [])
            new_todos = [t for t in todos if t.get("todo_uid") != todo_uid]
            if len(new_todos) == len(todos):
                return False
            data[stream_id] = new_todos
            await _save_all(data)
            return True

    async def set_reminder(self, stream_id: str, todo_uid: str, remind_at: float) -> bool:
        async with _lock:
            data = await _load_all()
            todos = data.get(stream_id, [])
            for t in todos:
                if t.get("todo_uid") == todo_uid:
                    t["remind_at"] = remind_at
                    t["reminded"] = False
                    data[stream_id] = todos
                    await _save_all(data)
                    return True
            return False

    async def clear_completed(self, stream_id: str) -> int:
        async with _lock:
            data = await _load_all()
            todos = data.get(stream_id, [])
            before = len(todos)
            new_todos = [t for t in todos if t.get("status") == "pending"]
            data[stream_id] = new_todos
            await _save_all(data)
            return before - len(new_todos)

    async def check_and_send_reminders(self) -> None:
        """扫描到期提醒，通过 LLM 生成拟人化提醒发送。"""
        async with _lock:
            data = await _load_all()
            now_ts = _now()
            due_items: list[tuple[str, dict[str, Any]]] = []

            for sid, todos in data.items():
                for t in todos:
                    if (
                        t.get("remind_at") is not None
                        and t["remind_at"] <= now_ts
                        and not t.get("reminded")
                        and t.get("status") == "pending"
                    ):
                        due_items.append((sid, t))

            if not due_items:
                return

        # 逐条通过 LLM 生成提醒消息并发送
        for sid, t in due_items:
            content = str(t.get("content", ""))
            try:
                msg = await self._build_reminder_message(content)
            except Exception as e:
                logger.warning(f"LLM 生成提醒失败: {e}，使用纯文本回退")
                msg = f"[提醒] {content}"

            ok = await send_api.send_text(msg, stream_id=sid)
            if ok:
                async with _lock:
                    data2 = await _load_all()
                    todos2 = data2.get(sid, [])
                    for t2 in todos2:
                        if t2.get("todo_uid") == t.get("todo_uid"):
                            t2["recurrence_executed_count"] = int(t2.get("recurrence_executed_count") or 0) + 1
                            if _recurrence_done(t2):
                                t2["status"] = "done"
                                t2["reminded"] = True
                            elif t2.get("recurrence_type"):
                                t2["reminded"] = False
                                t2["remind_at"] = _next_recurrence_time(t2)
                            else:
                                t2["reminded"] = True
                            break
                    data2[sid] = todos2
                    await _save_all(data2)
                logger.debug(f"已发送提醒: {t['todo_uid']} -> {sid}")

    # ── 内部方法 ────────────────────────────────────────────────────────────

    async def _update_status(self, stream_id: str, todo_uid: str, status: str) -> bool:
        async with _lock:
            data = await _load_all()
            todos = data.get(stream_id, [])
            for t in todos:
                if t.get("todo_uid") == todo_uid:
                    t["status"] = status
                    data[stream_id] = todos
                    await _save_all(data)
                    return True
            return False

    async def _build_reminder_message(self, todo_content: str) -> str:
        """通过 LLM 生成符合 bot 完整人设的拟人化提醒话术。"""
        persona = get_core_config().personality
        nickname = persona.nickname or "我"
        side = f"\n人格侧面：{persona.personality_side}" if persona.personality_side else ""
        identity_line = f"\n身份：{persona.identity}"
        bg = f"\n背景故事（不应主动复述）：{persona.background_story}" if persona.background_story else ""

        prompt = (
            f"你是 {nickname}。现在你需要提醒用户一个到期的待办事项。\n\n"
            f"你的核心人格：{persona.personality_core}{side}\n"
            f"{identity_line}{bg}\n"
            f"你的表达风格：{persona.reply_style}\n\n"
            f"待办事项：{todo_content}\n\n"
            f"请用一句简短、自然、符合你人设的话来提醒用户这件事该做了。"
            f"不要出现'提醒'这个词，要像随口一提一样自然。"
            f"禁止输出前缀、引号或解释，只输出提醒内容本身。"
        )

        try:
            model_set = get_model_set_by_task("utils_small")
        except Exception as exc:
            logger.debug(f"utils_small 模型集不可用，回退到 utils: {exc}")
            model_set = get_model_set_by_task("utils")

        request = create_llm_request(
            model_set=model_set,
            request_name="todo_plugin_remind",
        )
        request.add_payload(LLMPayload(ROLE.USER, Text(prompt)))

        response = await request.send(stream=False)
        await response
        return (response.message or f"[提醒] {todo_content}").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# BotTodoService — Bot 自我待办
# ═══════════════════════════════════════════════════════════════════════════════


async def _load_bot_all() -> dict[str, list[dict[str, Any]]]:
    """Load all bot todo partitions from plugin storage."""

    data = await storage_api.load_json(_STORE_NAME, _BOT_DATA_KEY)
    return data if data is not None else {}


async def _save_bot_all(data: dict[str, list[dict[str, Any]]]) -> None:
    """Persist all bot todo partitions to plugin storage."""

    await storage_api.save_json(_STORE_NAME, _BOT_DATA_KEY, data)


class BotTodoService(BaseService):
    """Bot 自我待办服务。

    Bot 可以在对话中计划自己未来要做的事，到期后走独立 LLM 生成行为。
    其他插件可通过注册表（registry.py）向执行 LLM 注入工具。
    其他插件通过 get_service("todo_plugin:service:bot_todo_service") 调用。
    """

    service_name: str = "bot_todo_service"
    service_description: str = "Bot 自我计划管理：记录自己的待办，到期 LLM 生成行为"
    version: str = "1.0.0"

    async def register_bot_tool(self, tool_cls: type) -> bool:
        """注册一个工具类到 bot 执行 LLM 的工具表。

        供其他插件通过 Service API 调用，替代直接 import registry 模块。
        """

        from .registry import register_bot_tool

        register_bot_tool(tool_cls)
        return True

    async def unregister_bot_tool(self, tool_cls: type) -> bool:
        """从 bot 执行工具表中移除一个工具类。"""

        from .registry import _bot_tools

        if tool_cls in _bot_tools:
            _bot_tools.remove(tool_cls)
            return True
        return False

    def _config(self) -> TodoPluginConfig:
        """Return todo config with safe defaults for direct service tests."""

        cfg = getattr(self.plugin, "config", None)
        return cfg if isinstance(cfg, TodoPluginConfig) else TodoPluginConfig()

    def _relay_retry_enabled(self) -> bool:
        """Return whether relay-origin bot plans should retry failed execution."""

        return bool(self._config().bot_execution.relay_retry_enabled)

    def _relay_max_retries(self) -> int:
        """Return non-negative max retry count for relay-origin bot plans."""

        return max(0, int(self._config().bot_execution.relay_max_retries))

    def _relay_retry_delay_seconds(self) -> int:
        """Return non-negative retry delay for relay-origin bot plans."""

        return max(0, int(self._config().bot_execution.relay_retry_delay_seconds))

    async def add_bot_todo(
        self,
        *,
        stream_id: str,
        plan: str,
        scheduled_at: float,
        recurrence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """记录一条 Bot 自己的待办计划。

        Args:
            stream_id: 来源聊天流（用于上下文）
            plan: bot 计划做什么
            scheduled_at: 计划执行时间戳
            recurrence: 循环规则，由 _parse_recurrence 解析得到
        """
        plan = plan.strip()
        if not plan:
            raise ValueError("计划内容不能为空")

        async with _bot_lock:
            data = await _load_bot_all()
            items: list[dict[str, Any]] = data.get(stream_id, [])

            item: dict[str, Any] = {
                "bot_todo_uid": _uid(),
                "plan": plan,
                "status": "pending",
                "scheduled_at": scheduled_at,
                "stream_id": stream_id,
                "created_at": _now(),
            }
            item.update(_build_recurrence_fields(recurrence))
            items.append(item)
            data[stream_id] = items
            await _save_bot_all(data)
            logger.debug(f"Bot 计划已记录: {stream_id} -> {item['bot_todo_uid']}")
            return item

    async def upsert_relay_todo(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        """Create or return a relay-origin bot task by conversation id."""

        conversation_id = str(payload.get("conversation_id") or "").strip()
        if not conversation_id:
            raise ValueError("conversation_id 不能为空")
        owner_bot = str(payload.get("owner_bot") or payload.get("to_bot") or "relay")
        stream_id = f"bot_relay:{owner_bot}"
        title = str(payload.get("title") or payload.get("summary") or "relay task").strip()
        scheduled_at = payload.get("due_at")
        try:
            scheduled_at = float(scheduled_at) if scheduled_at is not None else _now()
        except (TypeError, ValueError):
            scheduled_at = _now()

        async with _bot_lock:
            data = await _load_bot_all()
            items: list[dict[str, Any]] = data.get(stream_id, [])
            for item in items:
                if (
                    item.get("conversation_id") == conversation_id
                    and item.get("owner_bot") == owner_bot
                    and item.get("source") == "bot_private_relay"
                ):
                    logger.info(
                        "Relay Bot 计划已存在: "
                        f"conversation_id={conversation_id}, "
                        f"owner_bot={owner_bot}, "
                        f"stream_id={stream_id}, "
                        f"todo_uid={item.get('bot_todo_uid', '')}"
                    )
                    return {"ok": True, "todo_uid": item.get("bot_todo_uid", ""), "status": "duplicate", "item": item}

            item = {
                "bot_todo_uid": _uid(),
                "plan": title,
                "status": "pending",
                "scheduled_at": scheduled_at,
                "stream_id": stream_id,
                "created_at": _now(),
                "source": "bot_private_relay",
                "conversation_id": conversation_id,
                "trace_id": str(payload.get("trace_id") or ""),
                "owner_bot": owner_bot,
                "peer_bot_id": str(payload.get("peer_bot_id") or ""),
                "participants": list(payload.get("participants") or []),
                "source_stream_id": stream_id,
                "source_message_id": str(payload.get("source_message_id") or ""),
                "relay_decision": str(payload.get("decision") or ""),
                "summary": str(payload.get("summary") or title),
                "execution_channel": "social",
                "retry_count": int(payload.get("retry_count") or 0),
                "last_error": "",
                "last_attempt_at": None,
                "timed_out_at": None,
            }
            items.append(item)
            data[stream_id] = items
            await _save_bot_all(data)
            logger.info(
                "Relay Bot 计划已记录: "
                f"conversation_id={conversation_id}, "
                f"owner_bot={owner_bot}, "
                f"stream_id={stream_id}, "
                f"todo_uid={item['bot_todo_uid']}"
            )
            return {"ok": True, "todo_uid": item["bot_todo_uid"], "status": "created", "item": item}

    async def list_bot_todos(self, stream_id: str, status_filter: str = "") -> list[dict[str, Any]]:
        data = await _load_bot_all()
        items = data.get(stream_id, [])
        if status_filter and status_filter != "all":
            items = [t for t in items if t.get("status") == status_filter]
        return sorted(items, key=lambda t: t.get("scheduled_at", 0))

    async def list_bot_todos_for_streams(
        self,
        stream_ids: list[str],
        status_filter: str = "",
    ) -> list[dict[str, Any]]:
        """List bot plans from multiple isolated stream partitions."""

        data = await _load_bot_all()
        seen_streams: set[str] = set()
        items: list[dict[str, Any]] = []
        for stream_id in stream_ids:
            if not stream_id or stream_id in seen_streams:
                continue
            seen_streams.add(stream_id)
            items.extend(data.get(stream_id, []))
        if status_filter and status_filter != "all":
            items = [t for t in items if t.get("status") == status_filter]
        return sorted(items, key=lambda t: t.get("scheduled_at", 0))

    async def cancel_bot_todo(self, stream_id: str, todo_uid: str) -> bool:
        async with _bot_lock:
            data = await _load_bot_all()
            items = data.get(stream_id, [])
            for t in items:
                if t.get("bot_todo_uid") == todo_uid:
                    t["status"] = "cancelled"
                    data[stream_id] = items
                    await _save_bot_all(data)
                    return True
            return False

    async def check_and_execute_bot_tasks(self) -> None:
        """扫描到期的 bot 计划，通过 LLM（带注入工具）生成行为。"""
        async with _bot_lock:
            data = await _load_bot_all()
            now_ts = _now()
            due_items: list[tuple[str, dict[str, Any]]] = []

            for sid, items in data.items():
                for t in items:
                    if (
                        t.get("scheduled_at") is not None
                        and t["scheduled_at"] <= now_ts
                        and t.get("status") == "pending"
                    ):
                        due_items.append((sid, t))

            if not due_items:
                return

        for sid, t in due_items:
            plan = str(t.get("plan", ""))
            todo_uid = str(t.get("bot_todo_uid", ""))
            is_relay_plan = t.get("source") == "bot_private_relay"
            logger.info(
                "Bot 计划开始执行: "
                f"todo_uid={todo_uid}, stream_id={sid}, "
                f"source={t.get('source', 'manual')}, "
                f"conversation_id={t.get('conversation_id', '')}, "
                f"retry_count={int(t.get('retry_count') or 0)}"
            )
            try:
                result = await self._execute_bot_plan(plan, sid, task=t)
            except Exception as e:
                logger.warning(
                    "Bot 计划执行异常: "
                    f"todo_uid={todo_uid}, stream_id={sid}, error={e}"
                )
                result = BotPlanExecutionResult(ok=False, text=f"[计划执行失败] {plan}", error=str(e))

            if is_relay_plan:
                await self._record_relay_execution_result(sid=sid, todo_uid=todo_uid, result=result)
                continue

            await send_api.send_text(result.text, stream_id=sid)

            # 标记完成 / 重置循环
            async with _bot_lock:
                data2 = await _load_bot_all()
                items2 = data2.get(sid, [])
                for t2 in items2:
                    if t2.get("bot_todo_uid") == todo_uid:
                        t2["recurrence_executed_count"] = int(t2.get("recurrence_executed_count") or 0) + 1
                        t2["last_attempt_at"] = _now()
                        t2["last_error"] = "" if result.ok else result.error
                        if _recurrence_done(t2):
                            t2["status"] = "done"
                        elif t2.get("recurrence_type"):
                            t2["status"] = "pending"
                            t2["scheduled_at"] = _next_recurrence_time(t2)
                        else:
                            t2["status"] = "done"
                        break
                data2[sid] = items2
                await _save_bot_all(data2)

    async def _record_relay_execution_result(
        self,
        *,
        sid: str,
        todo_uid: str,
        result: BotPlanExecutionResult,
    ) -> None:
        """Persist relay-origin plan execution success, retry, or timeout."""

        now_ts = _now()
        async with _bot_lock:
            data = await _load_bot_all()
            items = data.get(sid, [])
            target: dict[str, Any] | None = None
            for item in items:
                if item.get("bot_todo_uid") == todo_uid:
                    target = item
                    break
            if target is None:
                logger.warning(f"Relay Bot 计划执行结果无法记录，计划不存在: todo_uid={todo_uid}, stream_id={sid}")
                return

            target["last_attempt_at"] = now_ts
            if result.ok:
                target["status"] = "done"
                target["last_error"] = ""
                target["timed_out_at"] = None
                logger.info(
                    "Relay Bot 计划执行完成: "
                    f"todo_uid={todo_uid}, stream_id={sid}, "
                    f"conversation_id={target.get('conversation_id', '')}, "
                    f"owner_bot={target.get('owner_bot', '')}, "
                    f"peer_bot_id={target.get('peer_bot_id', '')}"
                )
            else:
                retry_count = int(target.get("retry_count") or 0)
                max_retries = self._relay_max_retries()
                retry_enabled = self._relay_retry_enabled()
                error = result.error or "relay plan execution failed"
                target["last_error"] = error
                if retry_enabled and retry_count < max_retries:
                    next_retry_at = now_ts + self._relay_retry_delay_seconds()
                    target["retry_count"] = retry_count + 1
                    target["scheduled_at"] = next_retry_at
                    target["status"] = "pending"
                    logger.warning(
                        "Relay Bot 计划执行失败，将重试: "
                        f"todo_uid={todo_uid}, stream_id={sid}, "
                        f"conversation_id={target.get('conversation_id', '')}, "
                        f"owner_bot={target.get('owner_bot', '')}, "
                        f"peer_bot_id={target.get('peer_bot_id', '')}, "
                        f"attempt={retry_count + 1}/{max_retries + 1}, "
                        f"next_retry_count={retry_count + 1}, "
                        f"retry_after_seconds={self._relay_retry_delay_seconds()}, "
                        f"next_retry_at={next_retry_at}, "
                        f"error={error}"
                    )
                else:
                    target["status"] = "timeout"
                    target["timed_out_at"] = now_ts
                    logger.warning(
                        "Relay Bot 计划执行超时并放弃: "
                        f"todo_uid={todo_uid}, stream_id={sid}, "
                        f"conversation_id={target.get('conversation_id', '')}, "
                        f"owner_bot={target.get('owner_bot', '')}, "
                        f"peer_bot_id={target.get('peer_bot_id', '')}, "
                        f"attempts={retry_count + 1}, max_attempts={max_retries + 1}, "
                        f"retry_count={retry_count}, max_retries={max_retries}, "
                        f"retry_enabled={retry_enabled}, error={error}"
                    )
            data[sid] = items
            await _save_bot_all(data)

    async def _execute_bot_plan(
        self,
        plan: str,
        stream_id: str,
        task: dict[str, Any] | None = None,
    ) -> BotPlanExecutionResult:
        """Execute one bot plan through a todo-plugin-local LLM request."""

        from plugins.todo_plugin.registry import get_bot_tools

        task = task or {}
        is_relay_plan = task.get("source") == "bot_private_relay"
        owner_bot = str(task.get("owner_bot") or "")
        peer_bot_id = self._peer_bot_for_owner(task)
        required_tool_name = "tool-relay_social_contact"
        required_tool_called = False
        required_tool_succeeded = False
        required_tool_error = ""

        persona = get_core_config().personality
        nickname = persona.nickname or "bot"
        side = f"\n人格侧面：{persona.personality_side}" if persona.personality_side else ""
        bg = ""
        if not is_relay_plan and persona.background_story:
            bg = f"\n背景故事：{persona.background_story}"

        execution_rules = (
            f"你是 {nickname}。\n"
            f"核心人格：{persona.personality_core}{side}\n"
            f"身份：{persona.identity}{bg}\n"
            f"表达风格：{persona.reply_style}\n\n"
            "这是 todo_plugin 内部的 Bot 计划执行请求，不是普通聊天上下文。\n"
            "这些执行规则只适用于本次计划执行，不要外泄给用户或其他 bot。\n"
            "你可以调用可用工具来完成计划。"
            "最后，用一两句话自然地总结你做了什么；该总结只用于内部执行日志。"
        )
        execution_task = (
            f"你之前计划在这个时间做一件事：{plan}\n"
            "现在时间到了。请根据你的人设和这个计划行动。"
        )
        if is_relay_plan:
            execution_task += (
                "\n\n# relay 计划强制规则\n"
                "- 这条计划来自 bot_private_relay。\n"
                "- 你必须调用 relay_social_contact 工具完成对端联系，不允许只输出文本。\n"
                f"- target_bot_id 必须填写 peer_bot_id：{peer_bot_id or 'UNKNOWN'}。\n"
                f"- target_bot_id 禁止填写 owner_bot：{owner_bot or 'UNKNOWN'}。\n"
                "- reason 必须填写 todo_execution。\n"
                "- 只能通过 social channel 交流；禁止使用 transaction/system 或直接远程命令执行。\n"
                "- 你的最终文本不会直接发送给对端；真正外发必须通过 relay_social_contact。\n\n"
                "# relay 外发文案约束\n"
                "- relay_social_contact 的 message 由你生成，但只能基于 summary 和 plan 中已经确认的事实。\n"
                "- 不要引入新的任务、地点、人物关系、称呼、剧情、承诺、牺牲、守护、身份设定或情绪宣言。\n"
                "- 不要把对端误称为未在 summary/plan 出现的角色；对端只按 peer_bot_id 或已知对端称呼处理。\n"
                "- 如果 summary/plan 含糊，只发送一句简短澄清或确认，不要自由补全。\n"
                "- 文案保持一到两句话，语气可以符合人设，但事实范围必须严格受限。\n\n"
                "# relay 计划事实\n"
                f"- conversation_id: {task.get('conversation_id', '')}\n"
                f"- owner_bot: {owner_bot}\n"
                f"- peer_bot_id: {peer_bot_id}\n"
                f"- participants: {task.get('participants', [])}\n"
                f"- summary: {task.get('summary', '')}\n"
                f"- plan: {plan}\n"
            )

        try:
            model_set = get_model_set_by_task("utils_small")
        except Exception as exc:
            logger.debug(f"utils_small 模型集不可用，回退到 utils: {exc}")
            model_set = get_model_set_by_task("utils")

        request = create_llm_request(
            model_set=model_set,
            request_name="todo_plugin_bot_exec",
        )
        request.add_payload(LLMPayload(ROLE.SYSTEM, Text(execution_rules)))
        request.add_payload(LLMPayload(ROLE.USER, Text(execution_task)))

        # 注入其他插件注册的 Tool 类
        bot_tools: list[type] = []
        try:
            bot_tools = get_bot_tools()
        except Exception as exc:
            logger.debug(f"获取 bot tools 失败: {exc}")

        tool_map: dict[str, type] = {}
        for t in bot_tools:
            try:
                schema = t.to_schema()
                name = schema.get("function", schema).get("name", "")
                if name:
                    tool_map[name] = t
                    request.add_payload(LLMPayload(ROLE.TOOL, t))
            except Exception as exc:
                logger.debug(f"注入工具失败: {t}, error={exc}")

        logger.info(
            "Bot 计划 LLM 执行请求已构建: "
            f"stream_id={stream_id}, relay_plan={is_relay_plan}, "
            f"required_tool={required_tool_name if is_relay_plan else ''}, "
            f"tool_count={len(tool_map)}, owner_bot={owner_bot}, peer_bot_id={peer_bot_id}"
        )
        if is_relay_plan and required_tool_name not in tool_map:
            logger.warning(
                "Relay Bot 计划执行缺少必需工具: "
                f"required_tool={required_tool_name}, stream_id={stream_id}, "
                f"conversation_id={task.get('conversation_id', '')}"
            )

        # 第一轮 LLM
        response = await request.send(stream=False)
        await response
        # 同步 payloads，确保下一轮携带 assistant tool_call 消息
        request.payloads = list(response.payloads)

        # 处理 tool calls
        tool_calls = self._response_tool_calls(response)
        max_rounds = 3
        for _round in range(max_rounds):
            if not tool_calls:
                break

            logger.debug(f"Bot 计划 tool call 第{_round + 1}轮: {[c.name for c in tool_calls]}")

            for call in tool_calls:
                tool_cls = tool_map.get(call.name)
                if tool_cls is None:
                    request.add_payload(LLMPayload(
                        ROLE.TOOL_RESULT,
                        ToolResult(
                            value=f"工具未找到: {call.name}",
                            call_id=call.id,
                            name=call.name,
                        ),
                    ))
                    continue

                call_args: dict = dict(call.args) if isinstance(call.args, dict) else {}
                if is_relay_plan and call.name == required_tool_name:
                    required_tool_called = True
                    target_bot_id = str(call_args.get("target_bot_id") or "").strip()
                    if not peer_bot_id:
                        required_tool_error = "missing peer_bot_id for relay_social_contact"
                    elif target_bot_id != peer_bot_id:
                        required_tool_error = (
                            "relay_social_contact target_bot_id must equal peer_bot_id "
                            f"({peer_bot_id}), got {target_bot_id or '<empty>'}"
                        )
                    elif owner_bot and target_bot_id == owner_bot:
                        required_tool_error = "relay_social_contact target_bot_id must not equal owner_bot"
                    if required_tool_error:
                        logger.warning(
                            "Relay Bot 计划工具调用参数无效: "
                            f"stream_id={stream_id}, conversation_id={task.get('conversation_id', '')}, "
                            f"owner_bot={owner_bot}, peer_bot_id={peer_bot_id}, error={required_tool_error}"
                        )
                        request.add_payload(LLMPayload(
                            ROLE.TOOL_RESULT,
                            ToolResult(
                                value=f"执行失败: {required_tool_error}",
                                call_id=call.id,
                                name=call.name,
                            ),
                        ))
                        continue
                    call_args["target_bot_id"] = peer_bot_id
                    call_args["reason"] = "todo_execution"
                    call_args["conversation_id"] = str(task.get("conversation_id") or "")
                    call_args["trace_id"] = str(task.get("trace_id") or "")
                try:
                    tool_success, result = await exec_llm_usable(
                        tool_cls,
                        plugin=self.plugin,
                        stream_id=stream_id,
                        message=None,
                        kwargs=call_args,
                    )
                    result_text = str(result) if result else ""
                    if is_relay_plan and call.name == required_tool_name:
                        required_tool_succeeded = tool_success
                        if tool_success:
                            logger.info(
                                "Relay Bot 计划已通过 relay_social_contact 联系对端: "
                                f"stream_id={stream_id}, conversation_id={task.get('conversation_id', '')}, "
                                f"owner_bot={owner_bot}, peer_bot_id={peer_bot_id}"
                            )
                        else:
                            required_tool_error = result_text or "relay_social_contact failed"
                            logger.warning(
                                "Relay Bot 计划 relay_social_contact 执行失败: "
                                f"stream_id={stream_id}, conversation_id={task.get('conversation_id', '')}, "
                                f"owner_bot={owner_bot}, peer_bot_id={peer_bot_id}, error={required_tool_error}"
                            )
                except Exception as e:
                    logger.warning(f"执行工具 {call.name} 失败: {e}")
                    result_text = str(e)
                    if is_relay_plan and call.name == required_tool_name:
                        required_tool_succeeded = False
                        required_tool_error = str(e)

                request.add_payload(LLMPayload(
                    ROLE.TOOL_RESULT,
                    ToolResult(
                        value=result_text,
                        call_id=call.id,
                        name=call.name,
                    ),
                ))

                if is_relay_plan and required_tool_succeeded:
                    break

            if is_relay_plan and required_tool_succeeded:
                break

            # 继续 LLM 推理
            last_response = await request.send(stream=False)
            await last_response
            # 同步 payloads，确保下一轮携带 assistant tool_call 消息
            request.payloads = list(last_response.payloads)
            response = last_response
            tool_calls = self._response_tool_calls(response)

        final_text = (response.message or plan).strip()
        if is_relay_plan:
            if not required_tool_called:
                required_tool_error = "relay_social_contact was not called"
            ok = required_tool_called and required_tool_succeeded
            if not ok:
                logger.warning(
                    "Relay Bot 计划执行未满足强制工具调用要求: "
                    f"stream_id={stream_id}, conversation_id={task.get('conversation_id', '')}, "
                    f"owner_bot={owner_bot}, peer_bot_id={peer_bot_id}, "
                    f"called={required_tool_called}, succeeded={required_tool_succeeded}, "
                    f"error={required_tool_error}"
                )
            return BotPlanExecutionResult(
                ok=ok,
                text=final_text,
                required_tool_called=required_tool_called,
                required_tool_succeeded=required_tool_succeeded,
                error="" if ok else required_tool_error,
            )
        return BotPlanExecutionResult(ok=True, text=final_text)

    @staticmethod
    def _response_tool_calls(response: Any) -> list[Any]:
        """Return tool calls from either response API shape used in tests/runtime."""

        calls = getattr(response, "tool_calls", None)
        if calls is None:
            calls = getattr(response, "call_list", None)
        return list(calls or [])

    @staticmethod
    def _peer_bot_for_owner(task: dict[str, Any]) -> str:
        """Return the participant that is not owner_bot, preferring explicit peer."""

        owner_bot = str(task.get("owner_bot") or "").strip()
        explicit_peer = str(task.get("peer_bot_id") or "").strip()
        if explicit_peer and explicit_peer != owner_bot:
            return explicit_peer
        participants = task.get("participants")
        if isinstance(participants, list):
            for participant in participants:
                candidate = str(participant).strip()
                if candidate and candidate != owner_bot:
                    return candidate
        return explicit_peer
