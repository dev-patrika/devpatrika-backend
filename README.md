# Dev Patrika Backend Engine

An AI-powered Developer Intelligence Platform API Server built using **FastAPI**, **SQLModel** (SQLite database mapping), and the **LangChain** ecosystem.

## Project Structure (Phase 1 Layout)

```
app/
├── main.py             # Server bootstrapping and routing registration
├── config.py           # Settings management (pydantic-settings)
├── database.py         # SQLite connection setup
├── models/             # SQLModel DB tables (news, wiki, github_radar, notes)
├── schemas/            # Pydantic input/output schemas
├── routers/            # API Endpoints (/news, /wiki, /github, /search, /ai)
├── core/               # Shared constants, loggers, exceptions
├── services/           # Business logic packages (ingestion, processing, wiki_curator)
├── tools/              # Custom agent tools
├── agents/             # LangGraph agent definitions
└── prompts/            # Centralized prompt templates
```

## Getting Started (Local Development)

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Environment Configurations
Create a copy of `.env.example` as `.env` and configure local values:
```bash
cp .env.example .env
```

### 3. Local Installation
Create a virtual environment and install packages:
```bash
python -m venv venv
# On Windows PowerShell
.\venv\Scripts\Activate.ps1
# On Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Running the Server
Launch the development server with hot-reloads active:
```bash
uvicorn app.main:app --reload
```
Once initialized, visit:
- **API Status**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
