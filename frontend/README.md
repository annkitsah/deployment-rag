# Agentic Vectorless RAG — Frontend (Next.js / React)

React (Next.js App Router) frontend for the **Agentic True Vectorless RAG** system.

This replaces any Streamlit UI. It talks to the existing FastAPI backend over HTTP and is designed for deployment on **Vercel** (or Netlify, Cloudflare Pages, etc.).

## Features

- Upload PDF documents (ingested via backend OCR + lexical index)
- List ingested documents
- Ask questions (optionally scoped to a single document)
- View grounded answers with page-level citations and agent iteration count
- Health indicator showing indexed page count

## Prerequisites

- Node.js 18+
- A running FastAPI backend (see the backend repo / `docker/` folder)

## Local development

```bash
cd frontend   # or agentic-rag-frontend
cp .env.example .env.local
# Edit .env.local: set NEXT_PUBLIC_API_URL to your backend (e.g. http://localhost:8000)

npm install
npm run dev
```

Open http://localhost:3000

## Deploy on Vercel

1. Push this folder (or the whole monorepo) to GitHub/GitLab.
2. In Vercel: **New Project** → import the repo.
3. Set **Root Directory** to `frontend` (or whatever this folder is named) if it is not the repo root.
4. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = `https://your-backend-domain.example.com`
5. Deploy.

Also update the backend’s `CORS_ALLOWED_ORIGINS` (in `.env.production`) to include your Vercel URL, e.g.:

```
CORS_ALLOWED_ORIGINS=https://your-app.vercel.app,https://your-app-*.vercel.app
```

## API contract used

| Method | Path            | Purpose                          |
|--------|-----------------|----------------------------------|
| GET    | `/health`       | Health + indexed page count      |
| GET    | `/documents`    | List documents                   |
| POST   | `/documents`    | Upload PDF (`multipart/form-data`)|
| GET    | `/documents/{id}` | Get one document               |
| POST   | `/query`        | `{ question, document_id? }`     |

## Notes

- The backend does heavy work (OCR, indexing, LLM). The frontend is a thin client.
- Large PDF uploads may take minutes while the backend runs OCR + indexing.
- Queries can take 10–60+ seconds depending on model and iterations.
