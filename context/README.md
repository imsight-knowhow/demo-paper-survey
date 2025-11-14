# 项目上下文目录（context/）说明

## HEADER
- **Purpose**: 说明并规范 `context/` 目录的用途与结构，帮助人类与 AI 助手共享项目知识
- **Status**: Active
- **Date**: 2025-11-14
- **Dependencies**: magic-context/instructions/dir-setup/make-context-dir.md
- **Target**: 开发者与各类 AI 助手

## 内容说明

`context/` 目录是本项目的「知识与上下文中心」，用于存放与代码并行的文档、设计、日志和任务信息，方便多轮次、多人类与多 AI 协同开发。

本目录下的子目录约定如下：

- `design/`：系统与模块设计文档、接口规范等
- `hints/`：排错经验、How-To 指南、最佳实践
- `instructions/`：可复用的提示词片段、命令模板等
- `logs/`：开发过程日志与会话记录
- `plans/`：实现路线、版本规划与拆解方案
- `refcode/`：可复用/参考的代码片段与示例
- `roles/`：不同 AI 角色的系统提示与记忆
- `summaries/`：研究总结、知识沉淀与分析文档
- `tasks/`：任务看板（backlog / working / done）
- `tools/`：与开发流程相关的工具脚本说明

> 约定：本目录及其子目录中的文档，除非特别用英文标明，否则默认以中文撰写。

