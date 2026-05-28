"""Command routing tests for todo_plugin."""

from __future__ import annotations

import pytest

from plugins.todo_plugin.command import TodoCommand
from plugins.todo_plugin.config import TodoPluginConfig
from plugins.todo_plugin.plugin import TodoPlugin
from plugins.todo_plugin.service import BotTodoService, TodoService


def _install_command_services(monkeypatch, todo_service: TodoService, bot_service: BotTodoService) -> list[tuple[str, str]]:
    sent: list[tuple[str, str]] = []

    async def fake_send_text(text: str, stream_id: str) -> bool:
        sent.append((text, stream_id))
        return True

    def fake_get_service(signature: str):
        if signature == "todo_plugin:service:todo_service":
            return todo_service
        if signature == "todo_plugin:service:bot_todo_service":
            return bot_service
        return None

    class PluginManager:
        def get_plugin(self, name: str):
            return None

    monkeypatch.setattr("plugins.todo_plugin.command.send_api.send_text", fake_send_text)
    monkeypatch.setattr("plugins.todo_plugin.command.get_service", fake_get_service)
    monkeypatch.setattr("plugins.todo_plugin.command.get_plugin_manager", lambda: PluginManager())
    return sent


@pytest.mark.asyncio
async def test_todo_command_routes_user_todo_lifecycle(isolated_json_storage, monkeypatch) -> None:
    """The /todo command should route native user todo operations."""

    plugin = TodoPlugin(TodoPluginConfig())
    todo_service = TodoService(plugin)
    bot_service = BotTodoService(plugin)
    sent = _install_command_services(monkeypatch, todo_service, bot_service)
    command = TodoCommand(plugin=plugin, stream_id="stream-command")

    ok, result = await command.execute("add buy milk after work")
    assert ok is True
    assert result == "ok"
    todos = await todo_service.list_todos("stream-command", "all")
    assert [item["content"] for item in todos] == ["buy milk after work"]
    assert "buy milk after work" in sent[-1][0]

    uid = todos[0]["todo_uid"]
    ok, result = await command.execute("list all")
    assert ok is True
    assert result == "ok"
    assert uid in sent[-1][0]
    assert "buy milk after work" in sent[-1][0]
    assert "记录时间：" in sent[-1][0]

    ok, result = await command.execute(f"remind {uid} 30m")
    assert ok is True
    assert result == "ok"
    assert (await todo_service.list_todos("stream-command", "all"))[0]["remind_at"] is not None

    ok, result = await command.execute("list all")
    assert ok is True
    assert result == "ok"
    assert "提醒时间：" in sent[-1][0]

    ok, result = await command.execute(f"done {uid}")
    assert ok is True
    assert result == "ok"
    assert (await todo_service.list_todos("stream-command", "done"))[0]["todo_uid"] == uid

    ok, result = await command.execute(f"undo {uid}")
    assert ok is True
    assert result == "ok"
    assert (await todo_service.list_todos("stream-command", "pending"))[0]["todo_uid"] == uid

    ok, result = await command.execute(f"delete {uid}")
    assert ok is True
    assert result == "ok"
    assert await todo_service.list_todos("stream-command", "all") == []

    ok, result = await command.execute("unknown")
    assert ok is False
    assert "add, list" in result


@pytest.mark.asyncio
async def test_todo_command_clear_and_plans(isolated_json_storage, monkeypatch) -> None:
    """The /todo command should expose clear and native bot plan listing."""

    plugin = TodoPlugin(TodoPluginConfig())
    todo_service = TodoService(plugin)
    bot_service = BotTodoService(plugin)
    sent = _install_command_services(monkeypatch, todo_service, bot_service)
    command = TodoCommand(plugin=plugin, stream_id="stream-command")

    keep = await todo_service.add_todo(stream_id="stream-command", content="keep")
    done = await todo_service.add_todo(stream_id="stream-command", content="done")
    cancelled = await todo_service.add_todo(stream_id="stream-command", content="cancelled")
    assert keep is not None
    assert done is not None
    assert cancelled is not None
    await todo_service.mark_done("stream-command", done["todo_uid"])
    await todo_service.cancel_todo("stream-command", cancelled["todo_uid"])

    ok, result = await command.execute("clear")
    assert ok is True
    assert result == "ok"
    assert [item["content"] for item in await todo_service.list_todos("stream-command", "all")] == ["keep"]

    await bot_service.add_bot_todo(stream_id="stream-command", plan="check logs", scheduled_at=1.0)
    ok, result = await command.execute("plans")
    assert ok is True
    assert result == "ok"
    assert "check logs" in sent[-1][0]

    ok, result = await command.execute("list weird")
    assert ok is False
    assert result == "invalid status"


@pytest.mark.asyncio
async def test_todo_command_empty_root_shows_pending_list(isolated_json_storage, monkeypatch) -> None:
    """An empty /todo subroute should show pending todos for the current stream."""

    plugin = TodoPlugin(TodoPluginConfig())
    todo_service = TodoService(plugin)
    bot_service = BotTodoService(plugin)
    sent = _install_command_services(monkeypatch, todo_service, bot_service)
    command = TodoCommand(plugin=plugin, stream_id="stream-command")

    await todo_service.add_todo(stream_id="stream-command", content="root item")
    ok, result = await command.execute("")

    assert ok is True
    assert result == "ok"
    assert "root item" in sent[-1][0]
