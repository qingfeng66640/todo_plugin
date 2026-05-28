"""Native service tests for todo_plugin."""

from __future__ import annotations

import pytest

from plugins.todo_plugin.config import TodoPluginConfig
from plugins.todo_plugin.plugin import TodoPlugin
from plugins.todo_plugin.service import BotPlanExecutionResult, BotTodoService, TodoService


@pytest.mark.asyncio
async def test_user_todo_service_crud_filters_and_limits(isolated_json_storage) -> None:
    """TodoService should persist isolated user todos and enforce basic rules."""

    import src.app.plugin_system.api.storage_api as storage_api

    config = TodoPluginConfig()
    config.general.max_items_per_user = 2
    service = TodoService(TodoPlugin(config))

    first = await service.add_todo(stream_id="stream-a", content=" write tests ", priority=9)
    second = await service.add_todo(stream_id="stream-a", content="ship patch", priority=1)
    over_limit = await service.add_todo(stream_id="stream-a", content="extra item")
    other_stream = await service.add_todo(stream_id="stream-b", content="separate item")

    assert first is not None
    assert second is not None
    assert other_stream is not None
    assert first["content"] == "write tests"
    assert first["priority"] == 5
    assert second["priority"] == 1
    assert over_limit is None
    assert [item["todo_uid"] for item in await service.list_todos("stream-a")] == [
        first["todo_uid"],
        second["todo_uid"],
    ]
    assert [item["content"] for item in await service.list_todos("stream-b")] == ["separate item"]

    assert await service.mark_done("stream-a", second["todo_uid"]) is True
    assert await service.mark_done("stream-a", "missing") is False
    assert [item["todo_uid"] for item in await service.list_todos("stream-a", "done")] == [second["todo_uid"]]

    assert await service.mark_undo("stream-a", second["todo_uid"]) is True
    assert await service.cancel_todo("stream-a", second["todo_uid"]) is True
    assert await service.delete_todo("stream-a", first["todo_uid"]) is True
    assert await service.delete_todo("stream-a", first["todo_uid"]) is False

    stored = await storage_api.load_json("todo_plugin", "todos")
    assert stored is not None
    assert [item["status"] for item in stored["stream-a"]] == ["cancelled"]

    with pytest.raises(ValueError, match="不能为空"):
        await service.add_todo(stream_id="stream-a", content="   ")


@pytest.mark.asyncio
async def test_user_todo_service_reminders_mark_only_sent_items(isolated_json_storage, monkeypatch) -> None:
    """Due reminders should send once and persist the reminded flag after success."""

    import src.app.plugin_system.api.storage_api as storage_api

    service = TodoService(TodoPlugin(TodoPluginConfig()))
    due = await service.add_todo(stream_id="stream-remind", content="stand up", remind_at=1.0)
    future = await service.add_todo(stream_id="stream-remind", content="later", remind_at=9_999_999_999.0)
    done = await service.add_todo(stream_id="stream-remind", content="already done", remind_at=1.0)
    assert due is not None
    assert future is not None
    assert done is not None
    await service.mark_done("stream-remind", done["todo_uid"])

    async def fake_message(content: str) -> str:
        return f"time for {content}"

    sent: list[tuple[str, str]] = []

    async def fake_send_text(text: str, stream_id: str) -> bool:
        sent.append((text, stream_id))
        return True

    monkeypatch.setattr(service, "_build_reminder_message", fake_message)
    monkeypatch.setattr("plugins.todo_plugin.service.send_api.send_text", fake_send_text)

    await service.check_and_send_reminders()
    await service.check_and_send_reminders()

    assert sent == [("time for stand up", "stream-remind")]
    stored = await storage_api.load_json("todo_plugin", "todos")
    assert stored is not None
    by_uid = {item["todo_uid"]: item for item in stored["stream-remind"]}
    assert by_uid[due["todo_uid"]]["reminded"] is True
    assert by_uid[future["todo_uid"]]["reminded"] is False
    assert by_uid[done["todo_uid"]]["reminded"] is False


@pytest.mark.asyncio
async def test_clear_completed_keeps_pending_items(isolated_json_storage) -> None:
    """clear_completed should remove done and cancelled items without touching pending items."""

    service = TodoService(TodoPlugin(TodoPluginConfig()))
    pending = await service.add_todo(stream_id="stream-clear", content="keep")
    done = await service.add_todo(stream_id="stream-clear", content="done")
    cancelled = await service.add_todo(stream_id="stream-clear", content="cancelled")
    assert pending is not None
    assert done is not None
    assert cancelled is not None

    await service.mark_done("stream-clear", done["todo_uid"])
    await service.cancel_todo("stream-clear", cancelled["todo_uid"])

    assert await service.clear_completed("stream-clear") == 2
    assert [item["content"] for item in await service.list_todos("stream-clear", "all")] == ["keep"]


@pytest.mark.asyncio
async def test_bot_todo_service_native_lifecycle_and_due_execution(isolated_json_storage, monkeypatch) -> None:
    """BotTodoService should execute due non-relay plans and record their result."""

    import src.app.plugin_system.api.storage_api as storage_api

    service = BotTodoService(TodoPlugin(TodoPluginConfig()))
    success = await service.add_bot_todo(stream_id="stream-bot", plan="publish update", scheduled_at=1.0)
    failure = await service.add_bot_todo(stream_id="stream-bot", plan="fail update", scheduled_at=1.0)
    future = await service.add_bot_todo(stream_id="stream-bot", plan="future update", scheduled_at=9_999_999_999.0)

    assert [item["bot_todo_uid"] for item in await service.list_bot_todos("stream-bot")] == [
        success["bot_todo_uid"],
        failure["bot_todo_uid"],
        future["bot_todo_uid"],
    ]
    assert await service.cancel_bot_todo("stream-bot", "missing") is False
    assert await service.cancel_bot_todo("stream-bot", future["bot_todo_uid"]) is True
    assert [item["bot_todo_uid"] for item in await service.list_bot_todos("stream-bot", "pending")] == [
        success["bot_todo_uid"],
        failure["bot_todo_uid"],
    ]

    async def fake_execute(plan: str, stream_id: str, task=None) -> BotPlanExecutionResult:
        if plan == "fail update":
            return BotPlanExecutionResult(ok=False, text="could not run", error="boom")
        return BotPlanExecutionResult(ok=True, text=f"ran {plan}")

    sent: list[tuple[str, str]] = []

    async def fake_send_text(text: str, stream_id: str) -> bool:
        sent.append((text, stream_id))
        return True

    monkeypatch.setattr(service, "_execute_bot_plan", fake_execute)
    monkeypatch.setattr("plugins.todo_plugin.service.send_api.send_text", fake_send_text)

    await service.check_and_execute_bot_tasks()

    assert sent == [("ran publish update", "stream-bot"), ("could not run", "stream-bot")]
    stored = await storage_api.load_json("todo_plugin", "bot_todos")
    assert stored is not None
    by_uid = {item["bot_todo_uid"]: item for item in stored["stream-bot"]}
    assert by_uid[success["bot_todo_uid"]]["status"] == "done"
    assert by_uid[success["bot_todo_uid"]]["last_error"] == ""
    assert by_uid[failure["bot_todo_uid"]]["status"] == "done"
    assert by_uid[failure["bot_todo_uid"]]["last_error"] == "boom"
    assert by_uid[future["bot_todo_uid"]]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_bot_todo_service_lists_multiple_streams_once(isolated_json_storage) -> None:
    """Multi-stream listing should dedupe stream ids and sort by schedule time."""

    service = BotTodoService(TodoPlugin(TodoPluginConfig()))
    later = await service.add_bot_todo(stream_id="stream-one", plan="later", scheduled_at=20.0)
    earlier = await service.add_bot_todo(stream_id="stream-two", plan="earlier", scheduled_at=10.0)

    assert [item["bot_todo_uid"] for item in await service.list_bot_todos_for_streams([
        "stream-one",
        "stream-two",
        "stream-one",
    ])] == [earlier["bot_todo_uid"], later["bot_todo_uid"]]

    with pytest.raises(ValueError, match="不能为空"):
        await service.add_bot_todo(stream_id="stream-one", plan=" ", scheduled_at=1.0)
