# Demo Paper Survey

A demonstration project showcasing how to leverage AI agents to read, analyze academic papers, and conduct comprehensive topic surveys.

## Overview

This project demonstrates an automated pipeline for academic research assistance using AI agents. It covers:

- **Paper Reading & Parsing**: Extracting text and metadata from research papers (PDF, arXiv, etc.)
- **Intelligent Analysis**: Using AI agents to summarize, critique, and extract key insights
- **Topic Surveys**: Automated literature review and trend analysis across multiple papers
- **Knowledge Synthesis**: Aggregating findings and generating comprehensive survey reports

## Features (Planned)

### 1. Paper Ingestion
- PDF text extraction
- arXiv API integration
- Metadata extraction (authors, citations, publication date)
- Support for various academic paper formats

### 2. AI-Powered Analysis
- Paper summarization
- Key contribution extraction
- Methodology identification
- Results and conclusions analysis
- Citation network analysis

### 3. Topic Survey Generation
- Multi-paper comparative analysis
- Trend identification
- Research gap detection
- Automated survey report generation
- Visual knowledge graphs

### 4. Agent Orchestration
- Multi-agent workflows for different analysis tasks
- Parallel processing for multiple papers
- Iterative refinement of insights
- Human-in-the-loop feedback mechanisms

## Architecture (Proposed)

```
┌─────────────────┐
│ Paper Sources   │
│ (PDF, arXiv)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ingestion Agent │
│ - Parse & Extract│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analysis Agents │
│ - Summarize     │
│ - Extract Key   │
│ - Critique      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Survey Agent    │
│ - Compare       │
│ - Synthesize    │
│ - Generate      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Reports  │
│ - Markdown      │
│ - Visualizations│
└─────────────────┘
```

## Technology Stack (Planned)

- **Language**: Python 3.10+
- **AI Framework**: LangChain / LlamaIndex / AutoGen (TBD)
- **LLM Integration**: OpenAI API / Anthropic Claude / Local models
- **PDF Processing**: PyMuPDF, PDFPlumber
- **Data Storage**: Vector databases (Chroma, Pinecone)
- **Orchestration**: Agent frameworks for multi-agent coordination

## Project Structure (Proposed)

```
demo-paper-survey/
├── agents/              # AI agent definitions
├── ingestion/           # Paper parsing and extraction
├── analysis/            # Analysis modules
├── survey/              # Survey generation logic
├── utils/               # Helper functions
├── data/                # Sample papers and outputs
├── configs/             # Configuration files
├── notebooks/           # Jupyter notebooks for demos
└── tests/               # Unit and integration tests
```

## Getting Started

### Prerequisites
- Python 3.10 or higher
- API keys for LLM services (OpenAI, Anthropic, etc.)
- Virtual environment tool (venv, conda, or poetry)

### Installation
```bash
# Clone the repository
git clone https://github.com/imsight-knowhow/demo-paper-survey.git
cd demo-paper-survey

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (once requirements.txt is available)
pip install -r requirements.txt
```

### Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys and settings
```

## Usage Examples (Coming Soon)

### Analyze a Single Paper
```python
from agents import PaperAnalyzer

analyzer = PaperAnalyzer()
result = analyzer.analyze_paper("path/to/paper.pdf")
print(result.summary)
```

### Conduct Topic Survey
```python
from survey import TopicSurveyAgent

survey = TopicSurveyAgent(topic="transformer architectures")
report = survey.generate_survey(max_papers=20)
survey.save_report("output/survey_report.md")
```

## Roadmap

- [ ] Phase 1: Paper ingestion pipeline
- [ ] Phase 2: Single-paper analysis agent
- [ ] Phase 3: Multi-paper comparison
- [ ] Phase 4: Topic survey generation
- [ ] Phase 5: Interactive web interface
- [ ] Phase 6: Real-time arXiv monitoring

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

This is a demonstration project for educational purposes, showcasing the potential of AI agents in academic research assistance.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
