# Repository Guidelines

## Project Structure & Module Organization
- `src/demo_paper_survey/`: 核心 Python 包与业务逻辑。
- `paper-source/tex/`: 示例论文 StreamDiffusionV2 的 LaTeX 源文件与配套资产。
- `context/`: 项目知识库（设计、计划、日志、任务等），供人类与 AI 助手共享上下文。
- `magic-context/`: 上游子模块，提供通用指令与工具；通常不在此仓库内直接修改。
- `notes/`, `scripts/`: 试验性笔记与脚本（可按需扩展）。

## Build, Test, and Development Commands
- 开发环境（推荐使用 [pixi](https://pixi.sh/)）：  
  - `pixi shell` 进入虚拟环境。  
  - `pip install -e .` 在环境中以可编辑模式安装本包。
- 运行测试（添加测试后）：  
  - `pytest` 在仓库根目录执行全部测试。

## Coding Style & Naming Conventions
- 语言：Python 3.11+，遵循 PEP 8，4 空格缩进。
- 命名：模块与函数使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。
- 类型：优先使用类型注解与清晰的函数签名。
- 文档：除非有明确英文需求，README 与 `context/` 下文档优先使用中文。

## Testing Guidelines
- 测试框架：推荐使用 `pytest`。
- 目录结构：在仓库根添加 `tests/`，文件命名为 `test_*.py`。
- 要求：为新增的公共函数/类补充相应单元测试，覆盖主要分支与失败路径。

## Commit & Pull Request Guidelines
- 提交信息：使用简短的中文动词短语，描述本次主要变更，例如：`添加示例论文说明并下载 LaTeX 源文件`。
- 每次提交尽量只涵盖一组逻辑相关的改动（代码 + 测试 + 文档）。
- Pull Request：简要说明动机与变更范围，列出关键文件；如关联 Issue，请在描述中引用编号。

## AI Assistant & Context Usage
- 优先在 `context/` 中记录设计、计划、调研结果和已知坑点，便于 AI 在多轮会话中复用知识。
- 新的自动化流程或 Agent 行为说明，建议放在 `context/plans/` 或 `context/design/` 下并从主 `README.md` 进行链接。
