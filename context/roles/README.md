# roles/ 目录说明（角色与人设）

## HEADER
- **Purpose**: 定义不同 AI 助手角色的系统提示、职责边界与长期记忆
- **Status**: Active
- **Date**: 2025-11-14
- **Dependencies**: 上层 context/ 目录约定
- **Target**: AI 协作设计者、使用多角色协作的开发者

## 内容说明

`roles/` 目录中可以按照角色划分子目录，例如：

- `backend-developer/`
- `frontend-specialist/`
- `research-assistant/`

每个角色子目录中可以包含：

- `system-prompt.md`：该角色的系统提示与边界
- `memory.md`：该角色的长期知识与偏好
- `context.md` / `knowledge-base.md`：与角色相关的背景资料

