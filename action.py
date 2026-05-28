"""Todo Action — 供 LLM Tool Calling 使用。

Bot 可在对话中自然地为用户管理待办：
- add_user_todo：帮用户记录待办
- list_user_todos：查看用户待办列表
- mark_todo_done：标记待办完成

身份信息（stream_id）由框架 ChatStream 自动注入，LLM 无需关心"谁"创建了待办。

当 remind_at 为模糊时间描述（如"中午的时候"）时，通过 LLM 解析为具体时间戳。
"""

from __future__ import annotations

import datetime
import re
import time
from typing import Annotated, cast

from src.app.plugin_system.api.llm_api import create_llm_request, get_model_set_by_task
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.service_api import get_service
from src.core.components.base.action import BaseAction
from src.core.components.types import ChatType
from src.kernel.llm import LLMPayload, ROLE, Text

from .service import BotTodoService, TodoService

logger = get_logger("todo_plugin.action")


def _format_time(ts: float | None) -> str:
    """Format a timestamp for todo-facing action messages."""

    if ts is None:
        return "未知"
    return datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def _format_todo_time(todo: dict[str, object]) -> str:
    """Return the display time for a user todo."""

    remind_at = todo.get("remind_at")
    if isinstance(remind_at, (int, float)):
        return f"提醒时间：{_format_time(float(remind_at))}"

    created_at = todo.get("created_at")
    if isinstance(created_at, (int, float)):
        return f"记录时间：{_format_time(float(created_at))}"

    return "记录时间：未知"


async def _get_svc() -> TodoService:
    svc = get_service("todo_plugin:service:todo_service")
    if svc is None:
        raise RuntimeError("TodoService 未加载")
    return cast(TodoService, svc)


async def _get_bot_svc() -> BotTodoService:
    svc = get_service("todo_plugin:service:bot_todo_service")
    if svc is None:
        raise RuntimeError("BotTodoService 未加载")
    return cast(BotTodoService, svc)


def _parse_remind_time(raw: str) -> float | None:
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return None

    now = time.time()
    m = re.match(r"^(\d+)\s*(m|min|分钟?|h|小时?|d|天|s|秒)?$", raw, re.IGNORECASE)
    if m:
        value = int(m.group(1))
        unit = (m.group(2) or "m").lower()
        multipliers: dict[str, float] = {
            "s": 1, "秒": 1,
            "m": 60, "min": 60, "分钟": 60,
            "h": 3600, "小时": 3600,
            "d": 86400, "天": 86400,
        }
        return now + value * multipliers.get(unit, 60)

    try:
        parts = raw.split(":")
        if len(parts) == 2:
            hour, minute = int(parts[0]), int(parts[1])
            dt_now = datetime.datetime.now()
            dt = dt_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if dt <= dt_now:
                dt += datetime.timedelta(days=1)
            return dt.timestamp()
    except (ValueError, OverflowError):
        pass

    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%m-%d %H:%M"]:
        try:
            dt = datetime.datetime.strptime(raw, fmt)
            if "%Y" not in fmt:
                dt = dt.replace(year=datetime.datetime.now().year)
            return dt.timestamp()
        except ValueError:
            continue

    return None


def _has_time_hint(raw: str) -> bool:
    """Return whether text contains a natural-language time clue."""

    return bool(
        re.search(
            r"\d+\s*(m|min|分钟?|h|小时?|d|天|s|秒)|\d{1,2}:\d{2}|"
            r"今天|明天|明日|后天|早上|上午|中午|午饭|下午|晚上|今晚|明早|明晚|"
            r"周[一二三四五六日天]|星期[一二三四五六日天]|\d{1,2}\s*[点时]",
            raw,
            re.IGNORECASE,
        )
    )


def _day_offset(raw: str) -> int:
    """Infer a relative day offset from Chinese date words."""

    if "后天" in raw:
        return 2
    if any(word in raw for word in ("明天", "明日", "明早", "明晚")):
        return 1
    return 0


def _future_timestamp(hour: int, minute: int = 0, *, day_offset: int = 0) -> float:
    """Create a concrete future timestamp for a clock time."""

    now = datetime.datetime.now()
    dt = (now + datetime.timedelta(days=day_offset)).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    if day_offset == 0 and dt <= now:
        dt += datetime.timedelta(days=1)
    return dt.timestamp()


def _parse_common_fuzzy_time(raw: str) -> float | None:
    """Resolve common fuzzy Chinese time expressions without an LLM call."""

    text = raw.strip()
    if not text:
        return None

    day_offset = _day_offset(text)
    clock = re.search(r"(\d{1,2})\s*[点时](半|(?:(\d{1,2})\s*分?)?)", text)
    if clock:
        hour = int(clock.group(1))
        minute = 30 if clock.group(2) == "半" else int(clock.group(3) or 0)
        if any(word in text for word in ("下午", "晚上", "今晚", "明晚")) and hour < 12:
            hour += 12
        elif "中午" in text and hour < 11:
            hour += 12
        return _future_timestamp(hour, minute, day_offset=day_offset)

    default_times = [
        (("早上", "上午", "明早"), 8),
        (("中午", "午饭"), 12),
        (("下午",), 14),
        (("晚上", "今晚", "明晚"), 20),
    ]
    for words, hour in default_times:
        if any(word in text for word in words):
            return _future_timestamp(hour, day_offset=day_offset)

    if any(word in text for word in ("明天", "明日", "后天")):
        return _future_timestamp(9, day_offset=day_offset)

    return None


async def _resolve_fuzzy_time(fuzzy_text: str) -> float | None:
    """用 LLM 将模糊时间描述（如"中午的时候"、"明天早上"）解析为具体时间戳。

    Args:
        fuzzy_text: 用户/LLM 传递的模糊时间描述

    Returns:
        Unix 时间戳；解析失败返回 None
    """
    now_dt = datetime.datetime.now()
    weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
    prompt = (
        f"当前时间：{now_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        f"（星期{weekday_names[now_dt.weekday()]}）\n"
        f"用户描述的时间：{fuzzy_text}\n\n"
        f"请将用户的模糊时间描述解析为一个具体的时间戳。\n"
        f"规则：\n"
        f"- 中午/午饭 → 今天 12:00\n"
        f"- 下午 → 今天 14:00\n"
        f"- 晚上 → 今天 20:00\n"
        f"- 早上/上午 → 今天 8:00\n"
        f"- 明天X → 明天对应时间\n"
        f"- 一小时后 → 当前时间+3600\n"
        f"- N分钟后 → 当前时间+N*60\n"
        f"输出：只输出一个 Unix 时间戳数字（float），不要任何其他内容。"
    )

    try:
        model_set = get_model_set_by_task("utils_small")
    except Exception:
        model_set = get_model_set_by_task("utils")

    request = create_llm_request(
        model_set=model_set,
        request_name="todo_plugin_resolve_time",
    )
    request.add_payload(LLMPayload(ROLE.USER, Text(prompt)))

    try:
        response = await request.send(stream=False)
        await response
        raw = (response.message or "").strip()
        # 尝试直接解析为数字
        return float(raw)
    except (ValueError, TypeError):
        # 提取数字
        m = re.search(r"[\d.]+", raw)
        if m:
            try:
                return float(m.group())
            except ValueError:
                pass
    except Exception:
        pass
    return None


async def _resolve_action_time(time_text: str | None, fallback_text: str) -> float | None:
    """Resolve an explicit or inline action time into a concrete timestamp."""

    source = (time_text or "").strip()
    if source:
        parsed = _parse_remind_time(source) or _parse_common_fuzzy_time(source)
        return parsed if parsed is not None else await _resolve_fuzzy_time(source)

    if not _has_time_hint(fallback_text):
        return None

    parsed = _parse_common_fuzzy_time(fallback_text)
    return parsed if parsed is not None else await _resolve_fuzzy_time(fallback_text)


async def _resolve_required_action_time(time_text: str | None, fallback_text: str) -> float | None:
    """Resolve a required execution time, using the action text as fallback."""

    resolved = await _resolve_action_time(time_text, fallback_text)
    if resolved is not None:
        return resolved
    if _has_time_hint(fallback_text):
        return await _resolve_fuzzy_time(fallback_text)
    return None


class AddUserTodoAction(BaseAction):
    """添加用户待办事项。"""

    action_name: str = "add_user_todo"
    action_description: str = (
        "帮用户记录一条待提醒事项，到时间后提醒用户。"
        "当用户明确说'帮我记一下'、'提醒我XX'、'X分钟后叫我'时使用。"
    )
    primary_action: bool = False
    chat_type: ChatType = ChatType.ALL

    async def execute(
        self,
        content: Annotated[str, "用户需要被提醒的事项内容。保留用户原意，不要补充未提到的人名或对象"],
        priority: Annotated[int, "优先级 1-5，3 为默认"] = 3,
        remind_at: Annotated[str | None, "提醒时间。精确格式：'30m'/'3h'/'18:00'；也可留空，由事项内容中的'下午/晚上/明天'等时间线索自动推断"] = None,
    ) -> tuple[bool, str]:
        svc = await _get_svc()

        remind_ts = await _resolve_action_time(remind_at, content)

        item = await svc.add_todo(
            stream_id=self.chat_stream.stream_id,
            content=content,
            priority=priority,
            remind_at=remind_ts,
        )
        if item is None:
            return False, "待办事项已达上限，请先清理已完成的"

        detail = f"已记录待办 [{item['todo_uid']}]: {content}（{_format_todo_time(item)}）"
        return True, detail


class ListUserTodosAction(BaseAction):
    """查看用户待办列表。"""

    action_name: str = "list_user_todos"
    action_description: str = (
        "查看当前对话中用户让你帮忙记录的待办提醒列表（用户的事）。"
        "当用户问'我还有什么待办'、'帮我查一下待办'时使用。"
        "注意：这不会列出你自己的计划，你自己的计划用 list_bot_todos 查看。"
    )
    primary_action: bool = False
    chat_type: ChatType = ChatType.ALL

    async def execute(
        self,
        status: Annotated[str, "过滤状态：pending/done/all，默认 pending"] = "pending",
    ) -> tuple[bool, str]:
        svc = await _get_svc()

        filter_status = "" if status == "all" else (status if status in ("pending", "done") else "pending")
        todos = await svc.list_todos(self.chat_stream.stream_id, filter_status)

        if not todos:
            label = {"pending": "待处理", "done": "已完成", "": "全部"}.get(filter_status, "")
            return True, f"暂无{label}待办事项"

        lines = []
        for t in todos:
            uid = t.get("todo_uid", "")
            content = str(t.get("content", ""))
            s = t.get("status", "")
            mark = "✓" if s == "done" else "○"
            lines.append(f"  {mark} [{uid}] {content}（{_format_todo_time(t)}）")
        return True, "当前待办：\n" + "\n".join(lines)


class MarkTodoDoneAction(BaseAction):
    """标记待办完成。"""

    action_name: str = "mark_todo_done"
    action_description: str = (
        "标记当前对话用户的一条待办事项为已完成。"
        "当用户说'做完了'、'完成了'、'那个事搞定了'时使用。需要提供待办的 uid。"
    )
    primary_action: bool = False
    chat_type: ChatType = ChatType.ALL

    async def execute(
        self,
        todo_uid: Annotated[str, "待办事项的 uid（从 list_user_todos 结果中获取）"],
    ) -> tuple[bool, str]:
        svc = await _get_svc()

        ok = await svc.mark_done(self.chat_stream.stream_id, todo_uid.strip())
        if ok:
            return True, f"已完成待办: {todo_uid}"
        return False, f"未找到待办: {todo_uid}，请用 list_user_todos 确认 uid"


class ScheduleBotTaskAction(BaseAction):
    """Bot 为自己安排计划。"""

    action_name: str = "schedule_bot_task"
    action_description: str = (
        "记录你自己的日程安排——任何你（bot）未来要去参与或去做的事情。"
        "包括但不限于：和某人吃饭、参加会议、去群里聊天、主动打招呼、查看消息、发送通知等。"
        "当你提到任何未来时间点的活动时，用此动作记录下来。"
    )
    primary_action: bool = False
    chat_type: ChatType = ChatType.ALL

    async def execute(
        self,
        plan: Annotated[str, "你自己未来要做的事。用第一人称视角描述，不要写成替用户设置提醒的格式"],
        scheduled_at: Annotated[str | None, "执行时间。精确格式：'30m'/'3h'/'18:00'；可留空，由计划内容中的'下午/晚上/明天'等时间线索自动推断"] = None,
    ) -> tuple[bool, str]:
        svc = await _get_bot_svc()

        remind_ts = await _resolve_required_action_time(scheduled_at, plan)

        if remind_ts is None:
            return False, f"无法解析时间: {scheduled_at or plan}"

        item = await svc.add_bot_todo(
            stream_id=self.chat_stream.stream_id,
            plan=plan,
            scheduled_at=remind_ts,
        )

        dt = datetime.datetime.fromtimestamp(remind_ts)
        return True, f"已记录我的计划 [{item['bot_todo_uid']}]: {plan}（{dt.strftime('%m-%d %H:%M')}执行）"


class ListBotTodosAction(BaseAction):
    """查看 Bot 自己的计划列表。"""

    action_name: str = "list_bot_todos"
    action_description: str = (
        "查看你自己（bot）之前安排的待执行计划。"
        "当你不确定自己有没有安排过某件事时使用。"
    )
    primary_action: bool = False
    chat_type: ChatType = ChatType.ALL

    async def execute(self) -> tuple[bool, str]:
        svc = await _get_bot_svc()
        todos = await svc.list_bot_todos(self.chat_stream.stream_id, "pending")

        if not todos:
            return True, "暂无待执行的个人计划"

        lines = ["我的待执行计划："]
        for t in todos:
            uid = t.get("bot_todo_uid", "")
            plan = str(t.get("plan", ""))
            ts = t.get("scheduled_at", 0)
            dt_str = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "未知"
            lines.append(f"  ○ [{uid}] {plan}（{dt_str}）")
        return True, "\n".join(lines)
