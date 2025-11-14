# StreamDiffusionV2 技术调研与分析

基于 AI 多智能体的自动化论文分析项目，以 StreamDiffusionV2 为例，展示如何系统性地调研和分析实时扩散模型技术。

> 语言说明 / Language  
> 本项目主要面向中文用户。除非在文档中**特别用英文标明**，本仓库内的所有文档默认以**中文**撰写。

## 项目概览

本项目旨在构建一个自动化的学术研究辅助流水线，核心能力包括：

- **论文读取与解析**：从 PDF、arXiv 等来源提取正文、结构和元数据
- **智能分析**：利用 AI agent 进行摘要、评论、关键信息抽取
- **主题调研（Survey）**：对同一主题下多篇论文进行系统性梳理与对比
- **知识综合**：整合多篇论文结论，自动生成结构化调研报告

## 示例论文

为便于演示本项目的能力与评估整体流程，我们将优先以以下论文作为示例对象进行解析与调研：

- **StreamDiffusionV2: Customize your online diffusion demo, in around 10 lines of code**  
  对应代码仓库：<https://github.com/chenfengxu714/StreamDiffusionV2>  

该论文的 LaTeX 源文件将被下载并存放在本仓库的 `paper-source/tex/` 目录下，用于后续的解析、分析与自动化调研实验。