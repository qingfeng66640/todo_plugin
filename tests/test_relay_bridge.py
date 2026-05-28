"""Tests for relay todo bridge integration."""

from __future__ import annotations

import pytest

from src.kernel.llm import ToolCall

from plugins.todo_plugin.config import TodoPluginConfig
from plugins.todo_plugin.command import TodoCommand
from plugins.todo_plugin.event_handler import RelayTodoEventHandler
from plugins.todo_plugin.action import AddUserTodoAction, ScheduleBotTaskAction
from plugins.todo_plugin.plugin import TodoPlugin, sync_todo_board
from plugins.todo_plugin.service import BotPlanExecutionResult, BotTodoService
from src.core.prompt import get_system_reminder_store
from src.kernel.event import EventDecision


@pytest.mark.asyncio
async def test_relay_todo_upsert_is_idempotent_by_conversation_id_and_owner_bot(tmp_path, monkeypatch) -> None:
    """Relay-origin todos should use conversation_id + owner_bot as the idempotency key."""

    import src.app.plugin_system.api.storage_api as storage_api
    from src.kernel.storage import JSONStore

    store = JSONStore(str(tmp_path / "json"))
    monkeypatch.setattr(storage_api, "_get_plugin_json_store", lambda _name: store)

    service = BotTodoService(TodoPlugin())
    payload = {
        "conversation_id": "conv-1",
        "trace_id": "trace-1",
        "decision": "confirm",
        "owner_bot": "114514",
        "participants": ["223123", "114514"],
        "title": "整理会议纪要",
        "summary": "整理会议纪要",
        "source_stream_id": "bot_relay:114514",
    }
    first = await service.upsert_relay_todo(payload=payload)
    second = await service.upsert_relay_todo(payload=payload)
    third = await service.upsert_relay_todo(payload={**payload, "owner_bot": "223123"})
    bot_data = await storage_api.load_json("todo_plugin", "bot_todos")
    user_data = await storage_api.load_json("todo_plugin", "todos")

    assert first["ok"] is True
    assert first["status"] == "created"
    assert second["ok"] is True
    assert second["status"] == "duplicate"
    assert second["todo_uid"] == first["todo_uid"]
    assert third["ok"] is True
    assert third["status"] == "created"
    assert third["todo_uid"] != first["todo_uid"]
    assert user_data in (None, {})
    assert bot_data is not None
    item = bot_data["bot_relay:114514"][0]
    assert item["bot_todo_uid"] == first["todo_uid"]
    assert item["source"] == "bot_private_relay"
    assert item["scheduled_at"] <= item["created_at"]
    assert "priority" not in item
    assert bot_data["bot_relay:223123"][0]["owner_bot"] == "223123"


@pytest.mark.asyncio
async def test_relay_todo_upsert_uses_due_at_when_supplied(tmp_path, monkeypatch) -> None:
    """Relay-origin bot tasks should preserve explicit due_at as scheduled_at."""

    import src.app.plugin_system.api.storage_api as storage_api
    from src.kernel.storage import JSONStore

    store = JSONStore(str(tmp_path / "json"))
    monkeypatch.setattr(storage_api, "_get_plugin_json_store", lambda _name: store)

    service = BotTodoService(TodoPlugin())
    result = await service.upsert_relay_todo(
        payload={
            "conversation_id": "conv-due",
            "owner_bot": "114514",
            "title": "准点联系对端",
            "due_at": 12345.0,
            "source_stream_id": "bot_relay:114514",
        }
    )

    assert result["ok"] is True
    assert result["item"]["scheduled_at"] == 12345.0


@pytest.mark.asyncio
async def test_todo_plans_lists_relay_owner_stream(tmp_path, monkeypatch) -> None:
    """/todo plans should include relay-origin bot plans stored under bot_relay:{owner_bot}."""

    import src.app.plugin_system.api.storage_api as storage_api
    from src.kernel.storage import JSONStore

    class RelaySection:
        bot_id = "114514"

    class RelayConfig:
        relay = RelaySection()

    class RelayPlugin:
        config = RelayConfig()

    class PluginManager:
        def get_plugin(self, name: str):
            return RelayPlugin() if name == "bot_private_relay" else None

    store = JSONStore(str(tmp_path / "json"))
    monkeypatch.setattr(storage_api, "_get_plugin_json_store", lambda _name: store)
    monkeypatch.setattr("plugins.todo_plugin.command.get_plugin_manager", lambda: PluginManager())

    sent: list[tuple[str, str]] = []

    async def fake_send_text(text: str, stream_id: str) -> bool:
        sent.append((text, stream_id))
        return True

    monkeypatch.setattr("plugins.todo_plugin.command.send_api.send_text", fake_send_text)

    plugin = TodoPlugin()
    service = BotTodoService(plugin)
    monkeypatch.setattr(
        "plugins.todo_plugin.command.get_service",
        lambda signature: service if signature == "todo_plugin:service:bot_todo_service" else None,
    )
    await service.upsert_relay_todo(
        payload={
            "conversation_id": "conv-plans",
            "owner_bot": "114514",
            "peer_bot_id": "2899373955",
            "title": "与 2899373955 确认的计划：吃饭",
            "source_stream_id": "bot_relay:114514",
        }
    )

    command = TodoCommand(plugin=plugin, stream_id="qq:private:user")
    ok, status = await command.handle_plans()

    assert ok is True
    assert status == "ok"
    assert sent
    assert sent[0][1] == "qq:private:user"
    assert "与 2899373955 确认的计划：吃饭" in sent[0][0]


@pytest.mark.asyncio
async def test_relay_todo_event_handler_updates_preallocated_result(monkeypatch) -> None:
    """Event handler must preserve EventBus param keys and update result in place."""

    class StubBotTodoService:
        async def upsert_relay_todo(self, *, payload):
            return {"ok": True, "todo_uid": "todo-1", "status": "created"}

    monkeypatch.setattr(
        "plugins.todo_plugin.event_handler.get_service",
        lambda signature: StubBotTodoService() if signature == "todo_plugin:service:bot_todo_service" else None,
    )
    handler = RelayTodoEventHandler(TodoPlugin())
    params = {
        "payload": {"conversation_id": "conv-1", "title": "整理会议纪要"},
        "result": {"ok": None, "todo_uid": "", "status": "", "error": ""},
    }
    keys = set(params.keys())
    decision, out = await handler.execute("bot_relay.todo_decided", params)

    assert decision == EventDecision.SUCCESS
    assert set(out.keys()) == keys
    assert out["result"]["ok"] is True
    assert out["result"]["todo_uid"] == "todo-1"
    assert out["result"]["status"] == "created"


def test_todo_action_schemas_do_not_include_stale_person_examples() -> None:
    """Tool schemas should not seed ordinary chat with concrete stale todo examples."""

    combined = str(AddUserTodoAction.to_schema()) + str(ScheduleBotTaskAction.to_schema())

    assert "qf" not in combined
    assert "提醒qf喝水" not in combined


@pytest.mark.asyncio
async def test_sync_todo_board_does_not_inject_stored_todo_content(tmp_path, monkeypatch) -> None:
    """The actor reminder must not leak concrete todo data from any stream."""

    import src.app.plugin_system.api.storage_api as storage_api
    from src.kernel.storage import JSONStore

    store = JSONStore(str(tmp_path / "json"))
    monkeypatch.setattr(storage_api, "_get_plugin_json_store", lambda _name: store)
    await storage_api.save_json(
        "todo_plugin",
        "todos",
        {
            "qq:private:other": [
                {
                    "todo_uid": "todo-1",
                    "content": "提醒qf喝水",
                    "status": "pending",
                    "priority": 3,
                    "remind_at": 12345.0,
                }
            ]
        },
    )
    await storage_api.save_json(
        "todo_plugin",
        "bot_todos",
        {
            "bot_relay:114514": [
                {
                    "bot_todo_uid": "bot-1",
                    "plan": "与 2899373955 确认的计划：吃饭",
                    "status": "pending",
                    "scheduled_at": 12345.0,
                }
            ]
        },
    )

    store_reminders = get_system_reminder_store()
    store_reminders.delete("actor", "待办事项看板")
    await sync_todo_board()
    reminder = store_reminders.get("actor", names=["待办事项看板"])

    assert "提醒qf喝水" not in reminder
    assert "与 2899373955 确认的计划：吃饭" not in reminder
    assert "按聊天流隔离保存" in reminder


@pytest.mark.asyncio
async def test_relay_plan_failure_retries_by_default(tmp_path, monkeypatch) -> None:
    """Relay bot plans should retry failed execution by default."""

    import src.app.plugin_system.api.storage_api as storage_api
    from src.kernel.storage import JSONStore

    store = JSONStore(str(tmp_path / "json"))
    monkeypatch.setattr(storage_api, "_get_plugin_json_store", lambda _name: store)

    plugin = TodoPlugin(TodoPluginConfig())
    service = BotTodoService(plugin)
    item = await service.upsert_relay_todo(
        payload={
            "conversation_id": "conv-retry",
            "owner_bot": "114514",
            "peer_bot_id": "223123",
            "title": "与 223123 确认的计划：吃饭",
        }
    )

    async def fail_plan(*_args, **_kwargs):
        return BotPlanExecutionResult(ok=False, text="", error="relay_social_contact was not called")

    monkeypatch.setattr(service, "_execute_bot_plan", fail_plan)
    await service.check_and_execute_bot_tasks()

    bot_data = await storage_api.load_json("todo_plugin", "bot_todos")
    stored = bot_data["bot_relay:114514"][0]
    assert item["status"] == "created"
    assert stored["status"] == "pending"
    assert stored["retry_count"] == 1
    assert stored["last_error"] == "relay_social_contact was not called"
    assert stored["last_attempt_at"] is not None


@pytest.mark.asyncio
async def test_relay_plan_failure_times_out_when_retry_disabled(tmp_path, monkeypatch) -> None:
    """Relay bot plans should timeout immediately when retry is disabled."""

    import src.app.plugin_system.api.storage_api as storage_api
    from src.kernel.storage import JSONStore

    store = JSONStore(str(tmp_path / "json"))
    monkeypatch.setattr(storage_api, "_get_plugin_json_store", lambda _name: store)

    config = TodoPluginConfig()
    config.bot_execution.relay_retry_enabled = False
    plugin = TodoPlugin(config)
    service = BotTodoService(plugin)
    await service.upsert_relay_todo(
        payload={
            "conversation_id": "conv-timeout",
            "owner_bot": "114514",
            "peer_bot_id": "223123",
            "title": "与 223123 确认的计划：吃饭",
        }
    )

    async def fail_plan(*_args, **_kwargs):
        return BotPlanExecutionResult(ok=False, text="", error="social_contact_send_failed")

    monkeypatch.setattr(service, "_execute_bot_plan", fail_plan)
    await service.check_and_execute_bot_tasks()

    bot_data = await storage_api.load_json("todo_plugin", "bot_todos")
    stored = bot_data["bot_relay:114514"][0]
    assert stored["status"] == "timeout"
    assert stored["retry_count"] == 0
    assert stored["last_error"] == "social_contact_send_failed"
    assert stored["timed_out_at"] is not None


@pytest.mark.asyncio
async def test_relay_plan_requires_relay_social_contact_tool(monkeypatch) -> None:
    """Relay plan execution should fail if the required social tool is not called."""

    config = TodoPluginConfig()
    plugin = TodoPlugin(config)
    service = BotTodoService(plugin)

    class FakeResponse:
        message = "我会去联系对方。"
        payloads = []
        tool_calls = []

        def __await__(self):
            async def _collect():
                return self.message

            return _collect().__await__()

    class FakeRequest:
        def __init__(self) -> None:
            self.payloads = []
            self.added_payloads = []

        def add_payload(self, payload):
            self.payloads.append(payload)
            self.added_payloads.append(payload)

        async def send(self, stream=False):
            return FakeResponse()

    fake_request = FakeRequest()
    monkeypatch.setattr("plugins.todo_plugin.service.get_model_set_by_task", lambda _task: [{}])
    monkeypatch.setattr("plugins.todo_plugin.service.create_llm_request", lambda **_kwargs: fake_request)
    monkeypatch.setattr("plugins.todo_plugin.registry.get_bot_tools", lambda: [])
    monkeypatch.setattr(
        "plugins.todo_plugin.service.get_core_config",
        lambda: type(
            "Core",
            (),
            {
                "personality": type(
                    "Personality",
                    (),
                    {
                        "nickname": "bot",
                        "personality_core": "可靠",
                        "personality_side": "",
                        "identity": "测试 bot",
                        "background_story": "",
                        "reply_style": "简洁",
                    },
                )()
            },
        )(),
    )

    result = await service._execute_bot_plan(
        "与 223123 确认的计划：吃饭",
        "bot_relay:114514",
        task={
            "source": "bot_private_relay",
            "conversation_id": "conv-required-tool",
            "owner_bot": "114514",
            "peer_bot_id": "223123",
            "participants": ["114514", "223123"],
            "summary": "吃饭",
        },
    )

    assert result.ok is False
    assert result.required_tool_called is False
    assert result.error == "relay_social_contact was not called"
    assert [payload.role.value for payload in fake_request.added_payloads[:2]] == ["system", "user"]
    prompt_text = "\n".join(
        getattr(part, "text", "")
        for payload in fake_request.added_payloads
        for part in payload.content
    )
    assert "relay_social_contact" in prompt_text
    assert "只能基于 summary 和 plan 中已经确认的事实" in prompt_text
    assert "不要引入新的任务、地点、人物关系、称呼、剧情、承诺、牺牲、守护、身份设定或情绪宣言" in prompt_text
    assert "如果 summary/plan 含糊，只发送一句简短澄清或确认，不要自由补全" in prompt_text
    assert "测试背景" not in prompt_text


@pytest.mark.asyncio
async def test_relay_plan_passes_transaction_ids_to_social_contact(monkeypatch) -> None:
    """Relay plan execution should keep the original transaction ids on social contact."""

    config = TodoPluginConfig()
    plugin = TodoPlugin(config)
    service = BotTodoService(plugin)

    class FakeResponse:
        def __init__(self, tool_calls=None, message="已联系对端。") -> None:
            self.message = message
            self.payloads = []
            self.tool_calls = tool_calls or []

        def __await__(self):
            async def _collect():
                return self.message

            return _collect().__await__()

    class FakeRequest:
        def __init__(self) -> None:
            self.payloads = []
            self.sent = 0

        def add_payload(self, payload):
            self.payloads.append(payload)

        async def send(self, stream=False):
            self.sent += 1
            if self.sent == 1:
                return FakeResponse([
                    ToolCall(
                        id="call-1",
                        name="tool-relay_social_contact",
                        args={"target_bot_id": "223123", "message": "喝一口水，放松眼睛。"},
                    )
                ])
            return FakeResponse([])

    class FakeRelaySocialContactTool:
        @classmethod
        def to_schema(cls):
            return {"type": "function", "function": {"name": "tool-relay_social_contact"}}

    captured_kwargs = []

    async def fake_exec_llm_usable(_tool_cls, *, plugin, stream_id, message, kwargs):
        captured_kwargs.append(dict(kwargs))
        return True, "ok"

    monkeypatch.setattr("plugins.todo_plugin.service.get_model_set_by_task", lambda _task: [{}])
    monkeypatch.setattr("plugins.todo_plugin.service.create_llm_request", lambda **_kwargs: FakeRequest())
    monkeypatch.setattr("plugins.todo_plugin.registry.get_bot_tools", lambda: [FakeRelaySocialContactTool])
    monkeypatch.setattr("plugins.todo_plugin.service.exec_llm_usable", fake_exec_llm_usable)
    monkeypatch.setattr(
        "plugins.todo_plugin.service.get_core_config",
        lambda: type(
            "Core",
            (),
            {
                "personality": type(
                    "Personality",
                    (),
                    {
                        "nickname": "bot",
                        "personality_core": "可靠",
                        "personality_side": "",
                        "identity": "测试 bot",
                        "background_story": "",
                        "reply_style": "简洁",
                    },
                )()
            },
        )(),
    )

    result = await service._execute_bot_plan(
        "与 223123 确认的计划：喝一口水",
        "bot_relay:114514",
        task={
            "source": "bot_private_relay",
            "conversation_id": "conv-social-link",
            "trace_id": "trace-social-link",
            "owner_bot": "114514",
            "peer_bot_id": "223123",
            "participants": ["114514", "223123"],
            "summary": "喝一口水",
        },
    )

    assert result.ok is True
    assert captured_kwargs == [
        {
            "target_bot_id": "223123",
            "message": "喝一口水，放松眼睛。",
            "reason": "todo_execution",
            "conversation_id": "conv-social-link",
            "trace_id": "trace-social-link",
        }
    ]
