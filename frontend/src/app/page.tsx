"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Document,
  listDocuments,
  uploadDocument,
  runQuery,
  getHealth,
  QueryResponse,
  API_URL,
} from "@/lib/api";
import { AnswerView } from "@/components/AnswerView";

type Tab = "query" | "documents";

export default function Home() {
  const [tab, setTab] = useState<Tab>("query");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  const [question, setQuestion] = useState("");
  const [querying, setQuerying] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  const [health, setHealth] = useState<{ status: string; indexed_pages: number } | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const refreshDocuments = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const data = await listDocuments();
      setDocuments(data.documents);
      setApiError(null);
    } catch (e) {
      setApiError(e instanceof Error ? e.message : "Failed to load documents");
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
    getHealth()
      .then((h) => setHealth({ status: h.status, indexed_pages: h.indexed_pages }))
      .catch(() => setHealth(null));
  }, [refreshDocuments]);

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported.");
      e.target.value = "";
      return;
    }

    const sizeMb = file.size / (1024 * 1024);

    // Hard limit: block and show popup if over 50 MB
    if (sizeMb > 50) {
      window.alert(
        `File size limit exceeded\n\n` +
          `This PDF is ${sizeMb.toFixed(1)} MB.\n` +
          `Maximum allowed size is 50 MB.\n\n` +
          `Please split the PDF or upload a smaller file.`
      );
      setUploadError(
        `File is ${sizeMb.toFixed(1)} MB and exceeds the 50 MB limit. Please upload a smaller PDF.`
      );
      e.target.value = "";
      return;
    }

    // Soft warning: 30–50 MB may be slow on free-tier OCR
    if (sizeMb > 30) {
      const ok = window.confirm(
        `Large PDF (${sizeMb.toFixed(1)} MB)\n\n` +
          `Files under 30 MB work most reliably.\n` +
          `This file may take longer and is more likely to hit OCR rate limits.\n\n` +
          `Continue with upload?`
      );
      if (!ok) {
        e.target.value = "";
        return;
      }
    }

    setUploading(true);
    setUploadError(null);
    try {
      const doc = await uploadDocument(file);
      await refreshDocuments();
      getHealth()
        .then((h) => setHealth({ status: h.status, indexed_pages: h.indexed_pages }))
        .catch(() => null);
      setSelectedDocId(doc.document_id);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    setQuerying(true);
    setQueryError(null);
    setResult(null);
    try {
      const res = await runQuery(question.trim(), selectedDocId);
      setResult(res);
    } catch (err) {
      setQueryError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setQuerying(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4 sm:px-6">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">
              Agentic Vectorless RAG
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Page-indexed retrieval · no embeddings required
            </p>
          </div>
          <div className="flex items-center gap-3 text-sm">
            {health ? (
              <span className="flex items-center gap-1.5 text-zinc-600 dark:text-zinc-300">
                <span
                  className={`inline-block h-2 w-2 rounded-full ${
                    health.status === "healthy" ? "bg-emerald-500" : "bg-amber-500"
                  }`}
                />
                {health.indexed_pages} pages indexed
              </span>
            ) : (
              <span className="text-zinc-400">API offline</span>
            )}
            <span className="hidden text-xs text-zinc-400 sm:inline" title={API_URL}>
              {API_URL.replace(/^https?:\/\//, "").slice(0, 28)}
            </span>
          </div>
        </div>
      </header>

      <div className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-5xl gap-1 px-4 sm:px-6">
          <button
            onClick={() => setTab("query")}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              tab === "query"
                ? "border-b-2 border-zinc-900 text-zinc-900 dark:border-zinc-100 dark:text-zinc-100"
                : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
            }`}
          >
            Ask a question
          </button>
          <button
            onClick={() => setTab("documents")}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              tab === "documents"
                ? "border-b-2 border-zinc-900 text-zinc-900 dark:border-zinc-100 dark:text-zinc-100"
                : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
            }`}
          >
            Documents ({documents.length})
          </button>
        </div>
      </div>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
        {apiError && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200">
            <strong>Backend unreachable.</strong> {apiError}
            <br />
            <span className="text-xs opacity-80">
              Set <code className="rounded bg-black/5 px-1">NEXT_PUBLIC_API_URL</code> to your
              FastAPI backend URL (e.g. https://your-api.example.com).
            </span>
          </div>
        )}

        {tab === "query" && (
          <div className="space-y-6">
            <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
              <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Scope (optional)
              </label>
              <select
                value={selectedDocId ?? ""}
                onChange={(e) => setSelectedDocId(e.target.value || null)}
                className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
              >
                <option value="">All documents</option>
                {documents.map((d) => (
                  <option key={d.document_id} value={d.document_id}>
                    {d.filename} ({d.page_count} pages)
                  </option>
                ))}
              </select>
            </div>

            <form onSubmit={handleQuery} className="space-y-3">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about your documents…"
                rows={3}
                className="w-full resize-y rounded-xl border border-zinc-300 bg-white px-4 py-3 text-sm shadow-sm focus:border-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-200 dark:border-zinc-700 dark:bg-zinc-900 dark:focus:ring-zinc-700"
                disabled={querying}
              />
              <div className="flex items-center gap-3">
                <button
                  type="submit"
                  disabled={querying || !question.trim()}
                  className="rounded-lg bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
                >
                  {querying ? "Thinking…" : "Ask"}
                </button>
                {querying && (
                  <span className="text-sm text-zinc-500">
                    Running agentic retrieval loop…
                  </span>
                )}
              </div>
            </form>

            {queryError && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
                {queryError}
              </div>
            )}

            {result && (
              <AnswerView
                query={result.query}
                answer={result.answer}
                iterations={result.iterations}
                citations={[...result.citations]}
              />
            )}
          </div>
        )}

        {tab === "documents" && (
          <div className="space-y-6">
            <div className="rounded-xl border border-dashed border-zinc-300 bg-white p-6 text-center dark:border-zinc-700 dark:bg-zinc-900">
              <p className="mb-3 text-sm text-zinc-600 dark:text-zinc-400">
                Recommended: under 25 pages and 30 MB. Hard limit: 50 MB.
                Large files may hit rate limits or time out.
              </p>
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white">
                {uploading ? "Uploading & ingesting…" : "Choose PDF"}
                <input
                  type="file"
                  accept=".pdf,application/pdf"
                  className="hidden"
                  onChange={handleUpload}
                  disabled={uploading}
                />
              </label>
              {uploadError && (
                <p className="mt-3 text-sm text-red-600 dark:text-red-400">{uploadError}</p>
              )}
            </div>

            <div className="rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center justify-between border-b border-zinc-100 px-4 py-3 dark:border-zinc-800">
                <h2 className="text-sm font-semibold">Ingested documents</h2>
                <button
                  onClick={refreshDocuments}
                  disabled={loadingDocs}
                  className="text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
                >
                  Refresh
                </button>
              </div>
              {loadingDocs ? (
                <p className="px-4 py-8 text-center text-sm text-zinc-400">Loading…</p>
              ) : documents.length === 0 ? (
                <p className="px-4 py-8 text-center text-sm text-zinc-400">
                  No documents yet. Upload a PDF to get started.
                </p>
              ) : (
                <ul className="divide-y divide-zinc-100 dark:divide-zinc-800">
                  {documents.map((d) => (
                    <li
                      key={d.document_id}
                      className="flex items-center justify-between gap-4 px-4 py-3"
                    >
                      <div className="min-w-0">
                        <p className="truncate font-medium">{d.filename}</p>
                        <p className="text-xs text-zinc-500">
                          {d.page_count} pages · {d.status} ·{" "}
                          {new Date(d.created_at).toLocaleString()}
                        </p>
                      </div>
                      <button
                        onClick={() => {
                          setSelectedDocId(d.document_id);
                          setTab("query");
                        }}
                        className="shrink-0 rounded-md border border-zinc-200 px-3 py-1.5 text-xs font-medium hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800"
                      >
                        Query this
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="border-t border-zinc-200 py-4 text-center text-xs text-zinc-400 dark:border-zinc-800">
        Agentic True Vectorless RAG · FastAPI backend + Next.js frontend
      </footer>
    </div>
  );
}