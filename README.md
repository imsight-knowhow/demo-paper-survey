# Demo Paper Survey（论文调研 Demo）

一个演示性质的项目，展示如何利用 AI 多智能体（agents）来阅读和分析学术论文，并自动化完成某一研究主题的系统性调研（survey）。

> 语言说明 / Language  
> 本项目主要面向中文用户。除非在文档中**特别用英文标明**，本仓库内的所有文档默认以**中文**撰写。

## 项目概览

本项目旨在构建一个自动化的学术研究辅助流水线，核心能力包括：

- **论文读取与解析**：从 PDF、arXiv 等来源提取正文、结构和元数据
- **智能分析**：利用 AI agent 进行摘要、评论、关键信息抽取
- **主题调研（Survey）**：对同一主题下多篇论文进行系统性梳理与对比
- **知识综合**：整合多篇论文结论，自动生成结构化调研报告

## 功能规划

### 1. 论文导入（Paper Ingestion）
- PDF 文本内容抽取
- arXiv API 集成
- 元数据提取（作者、引用、发表时间等）
- 支持多种学术论文格式

### 2. AI 驱动分析（AI-Powered Analysis）
- 论文自动摘要
- 关键贡献点抽取
- 方法与实验设定识别
- 结果与结论分析
- 引用与相关工作网络分析

### 3. 主题调研生成（Topic Survey Generation）
- 多论文对比分析
- 研究趋势识别
- 研究空白与未来方向挖掘
- 自动生成调研报告（survey）
- 结构化知识图谱 / 可视化输出

### 4. Agent 编排与协作（Agent Orchestration）
- 不同任务的多智能体工作流设计
- 多篇论文的并行处理
- 迭代式结论优化与重写
- 人类在环（Human-in-the-loop）反馈机制

## 架构示意（拟定）

```
┌─────────────────┐
│ 论文来源        │
│ (PDF, arXiv 等) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 导入 Agent       │
│ - 解析与抽取     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 分析 Agents     │
│ - 摘要          │
│ - 关键信息抽取  │
│ - 评论与对比    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 调研 Agent      │
│ - 多文献对比    │
│ - 结论综合      │
│ - 报告生成      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 输出报告        │
│ - Markdown      │
│ - 可视化图表    │
└─────────────────┘
```

## 技术栈规划

- **语言**：Python 3.10+
- **AI 框架**：LangChain / LlamaIndex / AutoGen（待最终确定）
- **大模型接入**：OpenAI API / Anthropic Claude / 本地模型
- **PDF 处理**：PyMuPDF、PDFPlumber 等
- **数据存储**：向量数据库（如 Chroma、Pinecone）
- **编排与调度**：多 Agent 协作框架

## 目录结构（拟定）

```
demo-paper-survey/
├── agents/              # AI Agent 定义
├── ingestion/           # 论文导入与解析
├── analysis/            # 分析模块
├── survey/              # 调研报告生成逻辑
├── utils/               # 工具与通用函数
├── data/                # 示例论文与输出结果
├── configs/             # 配置文件
├── notebooks/           # 演示用 Jupyter Notebook
└── tests/               # 单元与集成测试
```

## 快速开始

### 环境准备
- Python 3.10 或更高版本
- 至少一个大模型服务的 API Key（如 OpenAI、Anthropic 等）
- 虚拟环境工具（`venv` / Conda / Poetry 等）

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/imsight-knowhow/demo-paper-survey.git
cd demo-paper-survey

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖（在 requirements.txt 可用后）
pip install -r requirements.txt
```

### 配置说明
```bash
# 拷贝环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 API Key 与相关配置
```

## 使用示例（规划中）

### 分析单篇论文
```python
from agents import PaperAnalyzer

analyzer = PaperAnalyzer()
result = analyzer.analyze_paper("path/to/paper.pdf")
print(result.summary)
```

### 生成某一主题的调研报告
```python
from survey import TopicSurveyAgent

survey = TopicSurveyAgent(topic="transformer architectures")
report = survey.generate_survey(max_papers=20)
survey.save_report("output/survey_report.md")
```

## 开发路线（Roadmap）

- [ ] 阶段 1：论文导入与解析流水线
- [ ] 阶段 2：单篇论文分析 Agent
- [ ] 阶段 3：多论文对比分析
- [ ] 阶段 4：主题调研报告生成
- [ ] 阶段 5：交互式 Web 界面
- [ ] 阶段 6：实时 arXiv 监控与增量更新

## 贡献方式

欢迎任何形式的贡献！你可以：

- 提交 Issue 反馈问题或提出功能建议
- 提交 Pull Request 参与代码或文档改进
- 分享使用场景与需求，帮助我们优化 Agent 设计

## 开源许可

本项目采用 MIT License，具体条款见仓库中的 `LICENSE` 文件。

## 致谢

本项目主要用于教学与演示，旨在展示 AI 多智能体在学术研究辅助中的潜力和基本实现思路。

## 联系方式

如有问题或合作意向，欢迎在 GitHub 仓库中提交 Issue 进行交流。
