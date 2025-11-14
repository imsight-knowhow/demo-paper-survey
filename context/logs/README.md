# logs/ 目录说明（开发与会话日志）

## HEADER
- **Purpose**: 记录重要开发会话、决策过程与尝试结果，形成可追溯的历史
- **Status**: Active
- **Date**: 2025-11-14
- **Dependencies**: 上层 context/ 目录约定
- **Target**: 项目维护者、历史分析需要的 AI 助手

## 内容说明

`logs/` 目录用于保存带日期前缀的日志文件，例如：

- `YYYY-MM-DD_feature-name-implementation-success.md`
- `YYYY-MM-DD_bug-fix-attempt-failed.md`

每份日志建议包含：

- 背景与目标
- 尝试过的方案及结果（成功 / 失败）
- 关键决策与原因
- 后续改进建议

这样可以帮助后来者理解「为什么会是现在的实现方式」。

