# 待办事项助手

`todo_plugin` 是一个面向个人待办、提醒和机器人定时任务的 Neo-MoFox 插件。

## 功能

- 通过 `/待办` 或 `/todo` 命令管理个人待办。
- 通过 Action 组件让 LLM 在对话中添加、查询和完成待办。
- 支持自然语言时间提示，例如 `下午一起吃饭`、`明天早上提醒我`。
- 支持机器人任务计划，用于延迟执行机器人侧待办。

## 命令

```text
/待办 add <内容>
/待办 list [pending|done|all]
/待办 plans
/待办 done <uid>
/待办 undo <uid>
/待办 delete <uid>
/待办 remind <uid> <时间>
/待办 clear
/待办 help
```

## 作者

qf
