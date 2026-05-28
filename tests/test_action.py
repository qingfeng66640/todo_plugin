"""Action execution tests for todo_plugin."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from plugins.todo_plugin.action import (
    AddUserTodoAction,
    ListBotTodosAction,
    ListUserTodosAction,
    MarkTodoDoneAction,
    ScheduleBotTaskAction,
)
from plugins.todo_plugin.config import TodoPluginConfig
from plugins.todo_plugin.plugin import TodoPlugin
from plugins.todo_plugin.service import BotTodoService, TodoService


def _install_action_services(monkeypatch, todo_service: TodoService, bot_service: BotTodoService) -> None:
    def fake_get_service(signature: str):
        if signature == "todo_plugin:service:todo_service":
            return todo_service
        if signature == "todo_plugin:service:bot_todo_service":
            return bot_service
        return None

    monkeypatch.setattr("plugins.todo_plugin.action.get_service", fake_get_service)


def _chat_stream(stream_id: str = "stream-action"):
    return SimpleNamespace(stream_id=stream_id)


@pytest.mark.asyncio
async def test_user_todo_actions_manage_current_stream(isolated_json_storage, monkeypatch) -> None:
    """User todo actions should operate on the injected chat stream id."""

    plugin = TodoPlugin(TodoPluginConfig())
    todo_service = TodoService(plugin)
    bot_service = BotTodoService(plugin)
    _install_action_services(monkeypatch, todo_service, bot_service)
    stream = _chat_stream()

    add_action = AddUserTodoAction(chat_stream=stream, plugin=plugin)
    list_action = ListUserTodosAction(chat_stream=stream, plugin=plugin)
    mark_action = MarkTodoDoneAction(chat_stream=stream, plugin=plugin)

    ok, text = await add_action.execute("submit report", priority=9)
    assert ok is True
    assert "submit report" in text
    assert "记录时间：" in text
    todos = await todo_service.list_todos("stream-action", "all")
    assert len(todos) == 1
    assert todos[0]["priority"] == 5

    ok, text = await list_action.execute()
    assert ok is True
    assert todos[0]["todo_uid"] in text
    assert "submit report" in text
    assert "记录时间：" in text

    ok, text = await mark_action.execute(todos[0]["todo_uid"])
    assert ok is True
    assert todos[0]["todo_uid"] in text

    ok, text = await list_action.execute("done")
    assert ok is True
    assert "submit report" in text

    ok, text = await mark_action.execute("missing")
    assert ok is False
    assert "missing" in text


@pytest.mark.asyncio
async def test_add_user_todo_action_uses_fuzzy_time_resolution(isolated_json_storage, monkeypatch) -> None:
    """Fuzzy reminder text should be resolved before storing the user todo."""

    plugin = TodoPlugin(TodoPluginConfig())
    todo_service = TodoService(plugin)
    bot_service = BotTodoService(plugin)
    _install_action_services(monkeypatch, todo_service, bot_service)

    async def fake_resolve(_text: str) -> float:
        return 1_800_000_000.0

    monkeypatch.setattr("plugins.todo_plugin.action._resolve_fuzzy_time", fake_resolve)

    action = AddUserTodoAction(chat_stream=_chat_stream(), plugin=plugin)
    ok, text = await action.execute("drink water", remind_at="tomorrow morning")

    assert ok is True
    assert "drink water" in text
    assert "提醒时间：" in text
    stored = await todo_service.list_todos("stream-action", "all")
    assert stored[0]["remind_at"] == 1_800_000_000.0

    list_action = ListUserTodosAction(chat_stream=_chat_stream(), plugin=plugin)
    ok, text = await list_action.execute("all")
    assert ok is True
    assert "drink water" in text
    assert "提醒时间：" in text


@pytest.mark.asyncio
async def test_add_user_todo_action_infers_fuzzy_time_from_content(isolated_json_storage, monkeypatch) -> None:
    """A todo with an inline fuzzy time should be stored with a concrete reminder time."""

    plugin = TodoPlugin(TodoPluginConfig())
    todo_service = TodoService(plugin)
    bot_service = BotTodoService(plugin)
    _install_action_services(monkeypatch, todo_service, bot_service)

    action = AddUserTodoAction(chat_stream=_chat_stream(), plugin=plugin)
    ok, text = await action.execute("下午一起吃个饭")

    assert ok is True
    assert "提醒时间：" in text
    stored = await todo_service.list_todos("stream-action", "all")
    assert stored[0]["remind_at"] is not None

    list_action = ListUserTodosAction(chat_stream=_chat_stream(), plugin=plugin)
    ok, text = await list_action.execute("all")
    assert ok is True
    assert "下午一起吃个饭" in text
    assert "提醒时间：" in text


@pytest.mark.asyncio
async def test_bot_todo_actions_schedule_and_list_native_plans(isolated_json_storage, monkeypatch) -> None:
    """Bot todo actions should schedule and list the bot's own plans."""

    plugin = TodoPlugin(TodoPluginConfig())
    todo_service = TodoService(plugin)
    bot_service = BotTodoService(plugin)
    _install_action_services(monkeypatch, todo_service, bot_service)
    stream = _chat_stream()

    schedule_action = ScheduleBotTaskAction(chat_stream=stream, plugin=plugin)
    list_action = ListBotTodosAction(chat_stream=stream, plugin=plugin)

    ok, text = await schedule_action.execute("check the queue", "30m")
    assert ok is True
    assert "check the queue" in text
    stored = await bot_service.list_bot_todos("stream-action", "pending")
    assert len(stored) == 1
    assert stored[0]["plan"] == "check the queue"

    ok, text = await schedule_action.execute("下午一起吃个饭")
    assert ok is True
    assert "下午一起吃个饭" in text
    stored = await bot_service.list_bot_todos("stream-action", "pending")
    assert stored[1]["scheduled_at"] is not None

    ok, text = await list_action.execute()
    assert ok is True
    assert stored[0]["bot_todo_uid"] in text
    assert "check the queue" in text

    async def fail_resolve(_text: str) -> None:
        return None

    monkeypatch.setattr("plugins.todo_plugin.action._resolve_fuzzy_time", fail_resolve)
    ok, text = await schedule_action.execute("bad time", "not-a-time")
    assert ok is False
    assert "not-a-time" in text
