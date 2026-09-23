"use client";

import { useMemo } from "react";
import type { Citation } from "@/lib/api";
import { API_URL, getStoredPassword } from "@/lib/api";
import { exportPdf, exportPpt, exportTxt, exportWord } from "@/lib/export";

type Props = {
  query: string;
  answer: string;
  iterations: number;
  citations: Citation[];
};

async function openCitation(documentId: string, pageNumber: number): Promise<void> {
  const headers: Record<string, string> = {};
  const password = getStoredPassword();
  if (password) headers["X-App-Password"] = password;

  const res = await fetch(`${API_URL}/documents/${documentId}/file`, { headers });
  if (!res.ok) {
    window.alert(`Could not open source PDF (${res.status})`);
    return;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  // Chrome/Edge PDF viewer supports #page=N
  window.open(`${url}#page=${pageNumber}`, "_blank", "noopener,noreferrer");
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export function AnswerView({ query, answer, iterations, citations }: Props) {
  const html = useMemo(() => renderMarkdown(answer), [answer]);

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
            Answer
          </h2>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-zinc-400">
              {iterations} iteration{iterations !== 1 ? "s" : ""}
            </span>
            <ExportButton label="PDF" onClick={() => exportPdf(answer, query, citations)} />
            <ExportButton label="Word" onClick={() => exportWord(answer, query, citations)} />
            <ExportButton label="PPT" onClick={() => void exportPpt(answer, query, citations)} />
            <ExportButton label="TXT" onClick={() => exportTxt(answer, query, citations)} />
          </div>
        </div>

        <div
          className="answer-prose text-[15px] leading-relaxed text-zinc-800 dark:text-zinc-100"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>

      {citations.length > 0 && (
        <div className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
            Citations
          </h2>
          <p className="mb-3 text-xs text-zinc-500">
            Click a citation to open the source PDF at that page.
          </p>
          <ul className="space-y-2">
            {citations.map((c, i) => (
              <li key={`${c.document_id}-${c.page_number}-${i}`}>
                <button
                  type="button"
                  onClick={() => void openCitation(c.document_id, c.page_number)}
                  className="flex w-full items-start gap-3 rounded-lg bg-zinc-50 px-3 py-2 text-left text-sm transition hover:bg-zinc-100 dark:bg-zinc-950 dark:hover:bg-zinc-900"
                >
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-zinc-200 text-xs font-medium dark:bg-zinc-800">
                    {i + 1}
                  </span>
                  <div>
                    <span className="font-medium text-emerald-700 underline-offset-2 hover:underline dark:text-emerald-400">
                      {c.filename ?? c.document_id}
                    </span>
                    <span className="text-zinc-500"> · page {c.page_number}</span>
                    <span className="ml-2 text-xs text-zinc-400">
                      score {c.score.toFixed(3)}
                    </span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <style jsx global>{`
        .answer-prose h1 {
          font-size: 1.35rem;
          font-weight: 700;
          margin: 1.1rem 0 0.5rem;
        }
        .answer-prose h2 {
          font-size: 1.15rem;
          font-weight: 650;
          margin: 1rem 0 0.45rem;
        }
        .answer-prose h3 {
          font-size: 1.05rem;
          font-weight: 600;
          margin: 0.85rem 0 0.35rem;
        }
        .answer-prose p {
          margin: 0.55rem 0;
        }
        .answer-prose ul {
          list-style: disc;
          padding-left: 1.35rem;
          margin: 0.5rem 0;
        }
        .answer-prose ol {
          list-style: decimal;
          padding-left: 1.35rem;
          margin: 0.5rem 0;
        }
        .answer-prose li {
          margin: 0.25rem 0;
        }
        .answer-prose code {
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 0.88em;
          background: rgba(0, 0, 0, 0.06);
          padding: 0.1rem 0.35rem;
          border-radius: 0.25rem;
        }
        .dark .answer-prose code {
          background: rgba(255, 255, 255, 0.08);
        }
        .answer-prose pre {
          background: rgba(0, 0, 0, 0.06);
          padding: 0.85rem 1rem;
          border-radius: 0.5rem;
          overflow-x: auto;
          margin: 0.75rem 0;
          font-size: 0.88rem;
        }
        .dark .answer-prose pre {
          background: rgba(255, 255, 255, 0.06);
        }
        .answer-prose pre code {
          background: transparent;
          padding: 0;
        }
        .answer-prose table {
          width: 100%;
          border-collapse: collapse;
          margin: 0.85rem 0;
          font-size: 0.92rem;
        }
        .answer-prose th,
        .answer-prose td {
          border: 1px solid rgba(128, 128, 128, 0.35);
          padding: 0.45rem 0.65rem;
          text-align: left;
          vertical-align: top;
        }
        .answer-prose th {
          background: rgba(0, 0, 0, 0.04);
          font-weight: 600;
        }
        .dark .answer-prose th {
          background: rgba(255, 255, 255, 0.06);
        }
        .answer-prose strong {
          font-weight: 650;
        }
        .answer-prose blockquote {
          border-left: 3px solid rgba(128, 128, 128, 0.4);
          padding-left: 0.85rem;
          margin: 0.75rem 0;
          opacity: 0.9;
        }
      `}</style>
    </div>
  );
}

function ExportButton({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-md border border-zinc-200 bg-white px-2.5 py-1 text-xs font-medium text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-200 dark:hover:bg-zinc-800"
    >
      {label}
    </button>
  );
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineFormat(s: string): string {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

export function renderMarkdown(md: string): string {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const out: string[] = [];
  let inUl = false;
  let inOl = false;
  let inCode = false;
  let codeBuf: string[] = [];
  let inTable = false;
  let tableRows: string[][] = [];

  const flushLists = () => {
    if (inUl) {
      out.push("</ul>");
      inUl = false;
    }
    if (inOl) {
      out.push("</ol>");
      inOl = false;
    }
  };

  const flushTable = () => {
    if (!inTable) return;
    if (tableRows.length) {
      out.push("<table><thead>");
      tableRows.forEach((row, ri) => {
        if (ri === 0) {
          out.push(
            "<tr>" +
              row.map((c) => `<th>${inlineFormat(c)}</th>`).join("") +
              "</tr></thead><tbody>"
          );
        } else if (!row.every((c) => /^[-:\s]+$/.test(c))) {
          out.push(
            "<tr>" + row.map((c) => `<td>${inlineFormat(c)}</td>`).join("") + "</tr>"
          );
        }
      });
      out.push("</tbody></table>");
    }
    inTable = false;
    tableRows = [];
  };

  const parseRow = (line: string): string[] => {
    const cells = line.split("|").map((c) => c.trim());
    let start = 0;
    let end = cells.length;
    if (cells[0] === "") start = 1;
    if (cells[cells.length - 1] === "") end = cells.length - 1;
    return cells.slice(start, end);
  };

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCode) {
        out.push(`<pre><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`);
        codeBuf = [];
        inCode = false;
      } else {
        flushLists();
        flushTable();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeBuf.push(line);
      continue;
    }

    if (line.trim().startsWith("|") && line.includes("|")) {
      flushLists();
      inTable = true;
      tableRows.push(parseRow(line));
      continue;
    }
    flushTable();

    if (/^\s*[-*]\s+/.test(line)) {
      if (!inUl) {
        flushLists();
        out.push("<ul>");
        inUl = true;
      }
      out.push(`<li>${inlineFormat(line.replace(/^\s*[-*]\s+/, ""))}</li>`);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      if (!inOl) {
        flushLists();
        out.push("<ol>");
        inOl = true;
      }
      out.push(`<li>${inlineFormat(line.replace(/^\s*\d+\.\s+/, ""))}</li>`);
      continue;
    }

    flushLists();

    if (/^###\s+/.test(line))
      out.push(`<h3>${inlineFormat(line.replace(/^###\s+/, ""))}</h3>`);
    else if (/^##\s+/.test(line))
      out.push(`<h2>${inlineFormat(line.replace(/^##\s+/, ""))}</h2>`);
    else if (/^#\s+/.test(line))
      out.push(`<h1>${inlineFormat(line.replace(/^#\s+/, ""))}</h1>`);
    else if (/^>\s?/.test(line))
      out.push(`<blockquote>${inlineFormat(line.replace(/^>\s?/, ""))}</blockquote>`);
    else if (line.trim() === "") out.push("");
    else out.push(`<p>${inlineFormat(line)}</p>`);
  }

  flushLists();
  flushTable();
  if (inCode) out.push(`<pre><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`);
  return out.join("\n");
}