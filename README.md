# Agentic True Vectorless RAG — Deployable Package

This package converts the original project to a **React (Next.js) frontend + FastAPI backend** architecture so the UI can be deployed on **Vercel** (or any static/SSR host) while the heavy RAG backend runs separately (e.g. Oracle Cloud Always Free + Docker, Railway, Fly.io, a VPS, etc.).

## Structure

```
agentic-true-vectorless-rag-deploy/
├── backend/          # Original FastAPI app (Python 3.13)
│   ├── app/          # Core application code
│   ├── docker/       # Docker Compose for API + Ollama + Caddy
│   ├── tests/
│   └── ...
└── frontend/         # New Next.js (React) UI
    ├── src/app/      # App Router pages
    ├── src/lib/api.ts
    └── README.md
```

## What changed

- **Frontend**: There was no working Streamlit UI in the source zip (only empty `ui/` placeholders). A full **Next.js + React + Tailwind** frontend was added that consumes the existing FastAPI endpoints.
- The backend was left intact; it already had CORS support and production notes aimed at a Vercel frontend.

## Quick start

### 1. Backend (local or server)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
# Set MISTRAL_API_KEY (required for OCR) and optionally Ollama settings
uvicorn app.main:app --reload --port 8000
```

Or use Docker (recommended for production) — see `backend/docker/DEPLOYMENT.md`.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open http://localhost:3000

### 3. Deploy frontend on Vercel

1. Push the repo (or just the `frontend/` folder) to GitHub.
2. Import in Vercel, set root directory to `frontend` if needed.
3. Environment variable: `NEXT_PUBLIC_API_URL=https://your-backend-domain.com`
4. On the backend, set `CORS_ALLOWED_ORIGINS` to your Vercel URL(s).

## Architecture overview

See the end of this conversation / the analysis section for full architecture, components built, and optimization opportunities.
