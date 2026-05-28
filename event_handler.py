"""Todo plugin event handlers."""

from __future__ import annotations

from typing import Any, cast

from src.app.plugin_system.api.service_api import get_service
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseEventHandler
from src.kernel.event import EventDecision

from .service import BotTodoService


logger = get_logger("todo_plugin.relay_bridge")


class RelayTodoEventHandler(BaseEventHandler):
    """Record relay-confirmed todos using conversation_id idempotency."""

    handler_name = "relay_todo_bridge"
    handler_description = "Listen for bot_relay todo decisions"
    weight = 50
    init_subscribe = ["bot_relay.todo_decided"]

    async def execute(self, event_name: str, params: dict[str, Any]) -> tuple[EventDecision, dict[str, Any]]:
        """Handle a relay todo decision without changing param keys."""

        payload = params.get("payload")
        result = params.get("result")
        if not isinstance(result, dict):
            return EventDecision.PASS, params
        if event_name != "bot_relay.todo_decided" or not isinstance(payload, dict):
            result.update({"ok": False, "todo_uid": "", "status": "invalid_payload", "error": "invalid relay todo event"})
            logger.warning("Relay todo event rejected: invalid payload")
            return EventDecision.SUCCESS, params

        conversation_id = str(payload.get("conversation_id") or "")
        logger.info(
            "Relay todo event received: "
            f"conversation_id={conversation_id}, "
            f"decision={payload.get('decision')}, "
            f"owner_bot={payload.get('owner_bot')}"
        )
        svc = get_service("todo_plugin:service:bot_todo_service")
        if svc is None:
            result.update({"ok": False, "todo_uid": "", "status": "bot_todo_service_unavailable", "error": "BotTodoService 未加载"})
            logger.warning(
                "Relay todo event rejected: BotTodoService unavailable, "
                f"conversation_id={conversation_id}"
            )
            return EventDecision.SUCCESS, params

        try:
            bridge_result = await cast(BotTodoService, svc).upsert_relay_todo(payload=payload)
        except Exception as exc:
            result.update({"ok": False, "todo_uid": "", "status": "todo_record_failed", "error": str(exc)})
            logger.warning(
                "Relay todo event failed: "
                f"conversation_id={conversation_id}, error={exc}"
            )
            return EventDecision.SUCCESS, params

        result.update(
            {
                "ok": bridge_result.get("ok") is True,
                "todo_uid": str(bridge_result.get("todo_uid") or ""),
                "status": str(bridge_result.get("status") or ""),
                "error": str(bridge_result.get("error") or ""),
            }
        )
        logger.info(
            "Relay todo event handled: "
            f"conversation_id={conversation_id}, "
            f"status={result.get('status')}, "
            f"todo_uid={result.get('todo_uid')}"
        )
        return EventDecision.SUCCESS, params
