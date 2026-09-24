"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Document,
  listDocuments,
  uploadDocument,
  deleteDocument,
  runQuery,
  getHealth,
  getDocumentProgress,
  setStoredPassword,
  clearStoredPassword,
  getStoredPassword,
  QueryResponse,
  DocumentProgress,
  API_URL,
} from "@/lib/api";
import { AnswerView } from "@/components/AnswerView";

type Tab = "query" | "documents";

type BatchItem = {
  name: string;
  status: "queued" | "uploading" | "done" | "error" | "skipped";
  message?: string;
};

export default function Home() {
  const [tab, setTab] = useState<Tab>("query");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [ingestProgress, setIngestProgress] = useState<DocumentProgress | null>(null);
  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);

  const [question, setQuestion] = useState("");
  const [querying, setQuerying] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  const [health, setHealth] = useState<{ status: string; indexed_pages: number } | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const [authed, setAuthed] = useState(false);
  const [passwordInput, setPasswordInput] = useState("");
  const [loginError, setLoginError] = useState<string | null>(null);

  const fetchHealth = useCallback(() => {
    getHealth()
      .then((h) => setHealth({ status: h.status, indexed_pages: h.indexed_pages }))
      .catch(() => setHealth(null));
  }, []);

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
    if (getStoredPassword()) setAuthed(true);
  }, []);

  useEffect(() => {
    if (!authed) return;
    refreshDocuments();
    fetchHealth();
  }, [authed, refreshDocuments, fetchHealth]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError(null);
    setStoredPassword(passwordInput);
    try {
      await listDocuments();
      setAuthed(true);
    } catch {
      clearStoredPassword();
      setLoginError("Wrong password");
      setAuthed(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files;
    if (!list || list.length === 0) return;

    const files = Array.from(list);
    e.target.value = "";

    const items: BatchItem[] = files.map((f) => ({
      name: f.name,
      status: "queued",
    }));
    setBatchItems(items);
    setUploadError(null);
    setIngestProgress(null);
    setUploading(true);

    const updateItem = (index: number, patch: Partial<BatchItem>) => {
      setBatchItems((prev) =>
        prev.map((it, i) => (i === index ? { ...it, ...patch } : it))
      );
    };

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];

        if (!file.name.toLowerCase().endsWith(".pdf")) {
          updateItem(i, { status: "skipped", message: "Not a PDF" });
          continue;
        }

        const sizeMb = file.size / (1024 * 1024);

        if (sizeMb > 50) {
          updateItem(i, {
            status: "skipped",
            message: `${sizeMb.toFixed(1)} MB > 50 MB limit`,
          });
          continue;
        }

        if (sizeMb > 30) {
          const ok = window.confirm(
            `Large PDF: ${file.name} (${sizeMb.toFixed(1)} MB)\n\n` +
              `Files under 30 MB are more reliable.\nContinue with this file?`
          );
          if (!ok) {
            updateItem(i, { status: "skipped", message: "Skipped by user" });
            continue;
          }
        }

        updateItem(i, { status: "uploading", message: "Uploading…" });

        try {
          const doc = await uploadDocument(file);
          setSelectedDocId(doc.document_id);
          updateItem(i, {
            status: "done",
            message: doc.duplicate
              ? "Already indexed (duplicate)"
              : `OK · ${doc.page_count} pages · ${doc.status}`,
          });
        } catch (err) {
          updateItem(i, {
            status: "error",
            message: err instanceof Error ? err.message : "Upload failed",
          });
        }
      }

      await refreshDocuments();
      fetchHealth();
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (documentId: string, filename: string) => {
    const ok = window.confirm(
      `Delete "${filename}"?\n\nThis removes the document and its indexed pages.`
    );
    if (!ok) return;

    try {
      await deleteDocument(documentId);
      if (selectedDocId === documentId) setSelectedDocId(null);
      await refreshDocuments();
      fetchHealth();
    } catch (err) {
      window.alert(err instanceof Error ? err.message : "Delete failed");
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

  const tabItems: { key: Tab; label: string }[] = [
    { key: "query", label: "Ask a question" },
    { key: "documents", label: `Documents (${documents.length})` },
  ];

  if (!authed) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-zinc-950">
        <form
          onSubmit={handleLogin}
          className="w-full max-w-sm rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
        >
          <h1 className="mb-1 text-lg font-semibold">Agentic Vectorless RAG</h1>
          <p className="mb-4 text-sm text-zinc-500">
            Enter the demo password to continue
          </p>
          <input
            type="password"
            value={passwordInput}
            onChange={(e) => setPasswordInput(e.target.value)}
            className="mb-3 w-full rounded-lg border border-zinc-300 px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
            placeholder="Password"
            autoFocus
          />
          {loginError && (
            <p className="mb-2 text-sm text-red-600">{loginError}</p>
          )}
          <button
            type="submit"
            className="w-full rounded-lg bg-zinc-900 py-2.5 text-sm font-medium text-white dark:bg-zinc-100 dark:text-zinc-900"
          >
            Unlock
          </button>
        </form>
      </div>
    );
  }

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
            <button
              type="button"
              onClick={() => {
                clearStoredPassword();
                setAuthed(false);
              }}
              className="text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
            >
              Logout
            </button>
            <span className="hidden text-xs text-zinc-400 sm:inline" title={API_URL}>
              {API_URL.replace(/^https?:\/\//, "").slice(0, 28)}
            </span>
          </div>
        </div>
      </header>

      <div className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto flex max-w-5xl gap-1 px-4 sm:px-6">
          {tabItems.map((item) => (
            <button
              key={item.key}
              onClick={() => setTab(item.key)}
              className={`px-4 py-3 text-sm font-medium transition-colors ${
                tab === item.key
                  ? "border-b-2 border-zinc-900 text-zinc-900 dark:border-zinc-100 dark:text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
        {apiError && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200">
            <strong>Backend unreachable.</strong> {apiError}
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
                Select multiple PDFs at once. Each file max 50 MB (30 MB recommended).
                Very large folders need async workers later.
              </p>
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white">
                {uploading ? "Uploading batch…" : "Choose PDFs (one or many)"}
                <input
                  type="file"
                  accept=".pdf,application/pdf"
                  multiple
                  className="hidden"
                  onChange={handleUpload}
                  disabled={uploading}
                />
              </label>

              {uploadError && (
                <p className="mt-3 text-sm text-red-600 dark:text-red-400">{uploadError}</p>
              )}

              {batchItems.length > 0 && (
                <ul className="mx-auto mt-4 max-w-lg space-y-1.5 text-left text-sm">
                  {batchItems.map((it, i) => (
                    <li
                      key={`${it.name}-${i}`}
                      className="flex items-start justify-between gap-2 rounded-md border border-zinc-200 px-3 py-2 dark:border-zinc-700"
                    >
                      <span className="min-w-0 truncate font-medium">{it.name}</span>
                      <span
                        className={
                          it.status === "done"
                            ? "shrink-0 text-xs text-emerald-600"
                            : it.status === "error" || it.status === "skipped"
                              ? "shrink-0 text-xs text-red-600"
                              : it.status === "uploading"
                                ? "shrink-0 text-xs text-amber-600"
                                : "shrink-0 text-xs text-zinc-500"
                        }
                      >
                        {it.status}
                        {it.message ? ` · ${it.message}` : ""}
                      </span>
                    </li>
                  ))}
                </ul>
              )}

              {ingestProgress && ingestProgress.status === "processing" && (
                <div className="mx-auto mt-4 max-w-md text-left">
                  <div className="mb-1 flex justify-between text-xs text-zinc-500">
                    <span>{ingestProgress.message}</span>
                    <span>{ingestProgress.percent}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                    <div
                      className="h-full rounded-full bg-emerald-500 transition-all duration-300"
                      style={{ width: `${ingestProgress.percent}%` }}
                    />
                  </div>
                </div>
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
                  No documents yet. Upload PDFs to get started.
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
                      <div className="flex shrink-0 items-center gap-2">
                        <button
                          onClick={() => {
                            setSelectedDocId(d.document_id);
                            setTab("query");
                          }}
                          className="rounded-md border border-zinc-200 px-3 py-1.5 text-xs font-medium hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800"
                        >
                          Query this
                        </button>
                        <button
                          onClick={() => handleDelete(d.document_id, d.filename)}
                          className="rounded-md border border-red-200 px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-50 dark:border-red-900 dark:text-red-300 dark:hover:bg-red-950/40"
                        >
                          Delete
                        </button>
                      </div>
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