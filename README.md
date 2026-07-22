# 🚀 Dev Patrika Backend Engine

An enterprise-grade, AI-powered Developer Intelligence Platform API Server built using **FastAPI**, **SQLModel (PostgreSQL)**, and **LangChain/LangGraph**. 

The engine crawls the developer ecosystem (Hacker News, Dev.to, arXiv, GitHub), processes news items asynchronously via stateful agents, maintains an auto-curated semantic wiki glossary, and provides a conversational Retrieval-Augmented Generation (RAG) assistant.

---

## 🏗️ System Architecture & Stack

* **API Core**: FastAPI (asynchronous, type-safe, built-in Swagger docs).
* **Database**: Neon Serverless Postgres (managed connection pooling, auto-scaling).
* **Vector Store**: Neon `pgvector` index integration.
* **Embeddings**: Hugging Face Cloud Inference API utilizing `BAAI/bge-small-en-v1.5` (384 dimensions) for zero-RAM cloud deployments.
* **AI Orchestration**: LangGraph StateGraphs for stateful, traceable multi-agent workflows.
* **Authentication**: JWT-based access/refresh token logic featuring:
  * Google OAuth 2.0
  * GitHub OAuth 2.0
  * Brevo-powered Passwordless Email OTP Login
* **Observability**: Direct LangSmith tracing configuration.

---

## 📂 Project Structure

```
app/
├── main.py             # Server bootstrapping and routing registration
├── config.py           # Settings management (Pydantic-Settings & dotenv)
├── database.py         # Neon Postgres connection engine with pooling settings
├── models/             # SQLModel DB models (news, wiki, github_radar, reports, auth, chat)
├── schemas/            # Pydantic input/output validation schemas
├── routers/            # API Endpoints (auth, news, wiki, github, search, ai, feedback, health)
├── core/               # Shared constants, loggers, exceptions, JWT security helpers
├── services/           # Ingestion, summarization, and vector retrieval services
├── tools/              # Custom agent tools (news search, paper summarizers, etc.)
├── agents/             # LangGraph agent definitions (DailyBrief, WikiCurator, ExplainWhy)
├── scripts/            # Migration scripts (e.g. pgvector 384-dimension migration)
└── prompts/            # Centralized system and worker prompt templates
```

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
|:---|:---|:---|
| **POST** | `/api/auth/register` | Register a new user with email and password |
| **POST** | `/api/auth/login` | Email + password login returning JWT tokens |
| **POST** | `/api/auth/otp/send` | Request a passwordless login OTP email |
| **POST** | `/api/auth/otp/verify` | Verify OTP and return JWT session tokens |
| **POST** | `/api/auth/refresh` | Silent access token renewal via Refresh Token |
| **GET** | `/api/auth/google/login` | Redirect endpoint for Google OAuth |
| **GET** | `/api/auth/github/login` | Redirect endpoint for GitHub OAuth |
| **GET** | `/api/news` | Paginated news feeds with query and category filters |
| **POST** | `/api/news/ingest` | Trigger background crawlers (Hacker News, Dev.to, arXiv, GitHub) |
| **POST** | `/api/news/process` | Trigger LangGraph processing pipeline for pending news |
| **GET** | `/api/wiki` | Term glossary lookup with autocomplete support |
| **GET** | `/api/wiki/{term}/timeline` | Dynamic technology evolution milestones |
| **GET** | `/api/github/trending` | GitHub trending projects radar with AI summary |
| **GET** | `/api/search` | Unified search (relational + pgvector semantic lookup) |
| **POST** | `/api/ai/chat` | Conversational RAG assistant with citation mappings |
| **POST** | `/api/feedback` | Send feedback directly to SMTP delivery (Brevo) |
| **GET** | `/api/health` | Service health status check |

---

## 🚀 Setup & Local Execution

### 1. Prerequisites
* Python 3.10+
* Neon Postgres instance with the `pgvector` extension enabled:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

### 2. Environment Configuration
Create a `.env` file in the `Backend/` directory based on `.env.example`:
```bash
cp .env.example .env
```
Fill in the database URI and LLM credentials:
* `DATABASE_URL`: Your Neon Pooled Postgres Connection String
* `GROQ_API_KEY`: Groq API key (Primary model)
* `GEMINI_API_KEY`: Google Gemini API key (Backup failover model)
* `HUGGINGFACE_API_KEY`: Hugging Face Token (Vector Embeddings)
* `BREVO_API_KEY` & `BREVO_SENDER_EMAIL`: SMTP and API keys for Email OTP verification
* `GOOGLE_CLIENT_ID` / `GITHUB_CLIENT_ID` (Optional): Social logins configuration

### 3. Installation
Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows PowerShell
.\venv\Scripts\Activate.ps1

# On Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Running Database Migrations
If starting fresh or migrating from an older schema, run the database migrations and embed utility:
```bash
# This creates tables and indexes on Neon Postgres
python -c "from app.database import init_db; init_db()"

# (Optional) Re-embed existing items if upgrading embedding dimensions
python -m app.scripts.migrate_embeddings
```

### 5. Start the API Server
Launch the development server with hot-reloads:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
* **Interactive OpenAPI/Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **API Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
