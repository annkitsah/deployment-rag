const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
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
  const res = await fetch(`${API_URL}/documents`, { cache: "no-store" });
  return handleResponse<DocumentListResponse>(res);
}

export async function uploadDocument(file: File): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/documents`, {
    method: "POST",
    body: form,
  });
  return handleResponse<Document>(res);
}

export async function getDocument(documentId: string): Promise<Document> {
  const res = await fetch(`${API_URL}/documents/${documentId}`, { cache: "no-store" });
  return handleResponse<Document>(res);
}

export async function runQuery(
  question: string,
  documentId?: string | null
): Promise<QueryResponse> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      document_id: documentId || null,
    }),
  });
  return handleResponse<QueryResponse>(res);
}

export { API_URL };
