# DeepResearch
### Multi-Agent Research Orchestration

DeepResearch was built to solve a specific problem: the modern web is noisy, making deep technical analysis difficult. Instead of just summarizing search results, this platform orchestrates a team of AI analysts to dig deeper. It uses a **Map-Reduce** pattern via LangGraph to ensure every topic is investigated from multiple specialized angles—technical, economic, and social.

---

## Why this exists

Most AI tools stop at surface-level summaries. DeepResearch is built for actual **Discovery**.

The core idea is to move beyond "reading" and start **interrogating**. The agents use recursive inquiry—similar to how a technical auditor works. They don't take general answers at face value; they follow up with targeted questions like *"How is this implemented?"* or *"What are the trade-offs?"*. This forces the system to ground every insight in cited, technical data.

---

## Live Demo
[**Launch DeepResearch Engine**](https://langgraph-based-multi-agent-research-system.streamlit.app/)

Check out the live deployment to see the agents in action!

---

## Technical Foundations

The engine is built on a stack designed for absolute execution stability and high-impact output:

### Orchestration & Logic
- **LangGraph**: At the heart of the system is a complex state-machine that manages long-running research cycles and parallel agent operations.
- **Resilience**: Integrated automatic **Retry Policies** and **Traffic Jitter** to ensure 100% execution stability across thousands of parallel model calls.
- **LangChain**: Provides the framework for robust model interactions, prompt management, and advanced tool integration.
- **Intelligence**: Powered by Novita AI (LLMs), offering high-reasoning capabilities tailored for technical analysis.

### Retrieval & Data
- **Tavily Search**: A specialized search engine built for AI agents that prioritizes technical and academic context over SEO-driven web content.
- **Wikipedia**: Integrated as a source of truth for encyclopedic grounding and background context.

### Professional Presentation
- **Streamlit**: A clean, responsive interface that provides real-time visibility into the research team's progress and the "human-in-the-loop" refinement stage.
- **Multi-Format Export**: Custom export logic for DOCX, PPTX, and PDF, ensuring your results are ready for professional presentation across any platform.

---

## Getting Started

### 1. Configuration
The system requires API credentials for Novita AI and Tavily. Initialize your local configuration by cloning the provided example:
```bash
cp .env.example .env
```
Add your personal API keys to the `.env` file before proceeding.

### 2. The Environment
We recommend using a virtual environment to manage dependencies:
```bash
python -m venv env
source env/bin/activate  
# Windows users: env\Scripts\activate
pip install -r requirements.txt
```

### 3. Launching the Engine
You can launch the Streamlit dashboard directly:
```bash
python -m streamlit run app.py
```
*Note: For Windows users, a convenience script `run_app.bat` is included which handles environment activation and requirement checks automatically.*

---

## Further Reading
For a comprehensive breakdown of the internal logic, agent states, and the Map-Reduce design pattern, refer to our [ARCHITECTURE.md](ARCHITECTURE.md).

---

*Sai Buvanesh*
