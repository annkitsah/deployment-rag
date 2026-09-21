const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const PASSWORD_KEY = "rag_app_password";

export function getStoredPassword(): string {
  if (typeof window === "undefined") return "";
  return sessionStorage.getItem(PASSWORD_KEY) || "";
}

export function setStoredPassword(password: string): void {
  sessionStorage.setItem(PASSWORD_KEY, password);
}

export function clearStoredPassword(): void {
  sessionStorage.removeItem(PASSWORD_KEY);
}

function authHeaders(extra?: HeadersInit): HeadersInit {
  const password = getStoredPassword();
  const headers: Record<string, string> = {};
  if (password) headers["X-App-Password"] = password;
  if (extra) Object.assign(headers, extra as Record<string, string>);
  return headers;
}

export interface Document {
  document_id: string;
  filename: string;
  page_count: number;
  status: string;
  created_at: string;
  duplicate?: boolean;
}

export interface DocumentListResponse {
  documents: Document[];
  count: number;
}

export interface Citation {
  document_id: string;
  filename: string | null;
  page_number: number;
  score: number;
}

export interface QueryResponse {
  query: string;
  answer: string;
  iterations: number;
  citations: Citation[];
}

export interface DocumentProgress {
  document_id: string;
  status: string;
  total_pages: number;
  processed_pages: number;
  percent: number;
  message: string;
  error: string | null;
}

export interface HealthResponse {
  status: string;
  service: string;
  environment: string;
  indexed_pages: number;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail)
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  return handleResponse<HealthResponse>(res);
}

export async function listDocuments(): Promise<DocumentListResponse> {
  const res = await fetch(`${API_URL}/documents`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  return handleResponse<DocumentListResponse>(res);
}

export async function uploadDocument(file: File): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/documents`, {
    method: "POST",
    body: form,
    headers: authHeaders(),
  });
  return handleResponse<Document>(res);
}

export async function getDocumentProgress(
  documentId: string
): Promise<DocumentProgress> {
  const res = await fetch(`${API_URL}/documents/${documentId}/progress`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  return handleResponse<DocumentProgress>(res);
}

export async function getDocument(documentId: string): Promise<Document> {
  const res = await fetch(`${API_URL}/documents/${documentId}`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  return handleResponse<Document>(res);
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await fetch(`${API_URL}/documents/${documentId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail)
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
}

export async function runQuery(
  question: string,
  documentId?: string | null
): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      question,
      document_id: documentId || null,
    }),
  });
  return handleResponse<QueryResponse>(res);
}

export { API_URL };