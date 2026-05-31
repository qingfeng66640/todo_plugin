# 使用说明

## 个人待办

用户可以通过 `/待办 add 买牛奶` 添加待办，也可以通过 `/待办 list` 查看未完成事项。

设置提醒时，插件支持相对时间、时钟时间和完整日期时间：

```text
/待办 remind abc123 30m
/待办 remind abc123 18:00
/待办 remind abc123 "2026-06-01 09:00"
```

## 对话 Action

LLM 可以调用以下 Action：

- `add_user_todo`
- `list_user_todos`
- `mark_todo_done`
- `schedule_bot_task`
- `list_bot_todos`

当待办内容包含明确时间线索时，Action 会尽量解析为具体时间戳；普通待办不会被强行附加提醒时间。
