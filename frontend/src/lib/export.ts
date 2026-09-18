import type { Citation } from "@/lib/api";

function citationsBlock(citations: Citation[]): string {
  if (!citations.length) return "";
  return (
    "\n\nCitations\n" +
    citations
      .map(
        (c, i) =>
          `${i + 1}. ${c.filename ?? c.document_id} — page ${c.page_number} (score ${c.score.toFixed(3)})`
      )
      .join("\n")
  );
}

function plainText(answer: string, query: string, citations: Citation[]): string {
  return `Question\n${query}\n\nAnswer\n${answer}${citationsBlock(citations)}\n`;
}

export function exportTxt(answer: string, query: string, citations: Citation[]) {
  const blob = new Blob([plainText(answer, query, citations)], {
    type: "text/plain;charset=utf-8",
  });
  downloadBlob(blob, "answer.txt");
}

export function exportWord(answer: string, query: string, citations: Citation[]) {
  const html = `<!DOCTYPE html>
<html xmlns:o="urn:schemas-microsoft-com:office:office"
      xmlns:w="urn:schemas-microsoft-com:office:word"
      xmlns="http://www.w3.org/TR/REC-html40">
<head><meta charset="utf-8"><title>Answer</title>
<style>
  body { font-family: Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.5; }
  h1 { font-size: 16pt; }
  h2 { font-size: 13pt; color: #333; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  th { background: #f3f4f6; }
</style>
</head>
<body>
  <h1>Agentic Vectorless RAG</h1>
  <h2>Question</h2>
  <p>${escapeHtml(query)}</p>
  <h2>Answer</h2>
  ${markdownToSimpleHtml(answer)}
  ${
    citations.length
      ? `<h2>Citations</h2><ol>${citations
          .map(
            (c) =>
              `<li>${escapeHtml(c.filename ?? c.document_id)} — page ${c.page_number} (score ${c.score.toFixed(3)})</li>`
          )
          .join("")}</ol>`
      : ""
  }
</body>
</html>`;

  const blob = new Blob(["\ufeff", html], {
    type: "application/msword;charset=utf-8",
  });
  downloadBlob(blob, "answer.doc");
}

export function exportPdf(answer: string, query: string, citations: Citation[]) {
  const html = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Answer</title>
<style>
  body { font-family: Georgia, serif; font-size: 12pt; line-height: 1.55; max-width: 720px; margin: 40px auto; color: #111; }
  h1 { font-size: 18pt; margin-bottom: 8px; }
  h2 { font-size: 13pt; margin-top: 24px; color: #333; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 11pt; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  th { background: #f3f4f6; }
  code { background: #f4f4f5; padding: 1px 4px; border-radius: 3px; }
  pre { background: #f4f4f5; padding: 12px; border-radius: 6px; overflow-x: auto; }
</style>
</head>
<body>
  <h1>Agentic Vectorless RAG</h1>
  <h2>Question</h2>
  <p>${escapeHtml(query)}</p>
  <h2>Answer</h2>
  ${markdownToSimpleHtml(answer)}
  ${
    citations.length
      ? `<h2>Citations</h2><ol>${citations
          .map(
            (c) =>
              `<li>${escapeHtml(c.filename ?? c.document_id)} — page ${c.page_number} (score ${c.score.toFixed(3)})</li>`
          )
          .join("")}</ol>`
      : ""
  }
  <script>window.onload = function(){ window.print(); }</script>
</body>
</html>`;

  const w = window.open("", "_blank");
  if (!w) {
    alert("Please allow pop-ups to export PDF.");
    return;
  }
  w.document.open();
  w.document.write(html);
  w.document.close();
}

export async function exportPpt(
  answer: string,
  query: string,
  citations: Citation[]
) {
  try {
    const PptxGenJS = (await import("pptxgenjs")).default;
    const pptx = new PptxGenJS();
    pptx.author = "Agentic Vectorless RAG";
    pptx.title = "RAG Answer";

    const title = pptx.addSlide();
    title.addText("Agentic Vectorless RAG", {
      x: 0.5, y: 1.5, w: 9, h: 0.6,
      fontSize: 28, bold: true, color: "111827",
    });
    title.addText(query, {
      x: 0.5, y: 2.4, w: 9, h: 1.5,
      fontSize: 16, color: "374151",
    });

    const chunks = splitForSlides(answer, 800);
    chunks.forEach((chunk, idx) => {
      const slide = pptx.addSlide();
      slide.addText(idx === 0 ? "Answer" : `Answer (cont. ${idx + 1})`, {
        x: 0.5, y: 0.3, w: 9, h: 0.4,
        fontSize: 14, bold: true, color: "111827",
      });
      slide.addText(chunk, {
        x: 0.5, y: 0.9, w: 9, h: 4.5,
        fontSize: 13, color: "1f2937", valign: "top",
      });
    });

    if (citations.length) {
      const cite = pptx.addSlide();
      cite.addText("Citations", {
        x: 0.5, y: 0.3, w: 9, h: 0.4,
        fontSize: 14, bold: true, color: "111827",
      });
      cite.addText(
        citations
          .map(
            (c, i) =>
              `${i + 1}. ${c.filename ?? c.document_id} — page ${c.page_number} (score ${c.score.toFixed(3)})`
          )
          .join("\n"),
        {
          x: 0.5, y: 0.9, w: 9, h: 4.5,
          fontSize: 13, color: "1f2937", valign: "top",
        }
      );
    }

    await pptx.writeFile({ fileName: "answer.pptx" });
  } catch {
    exportTxt(answer, query, citations);
    alert("PPT library missing — downloaded as text. Run: npm install pptxgenjs");
  }
}

function splitForSlides(text: string, maxLen: number): string[] {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= maxLen) return [clean];
  const parts: string[] = [];
  let rest = clean;
  while (rest.length > maxLen) {
    let cut = rest.lastIndexOf(". ", maxLen);
    if (cut < maxLen * 0.5) cut = maxLen;
    parts.push(rest.slice(0, cut + 1).trim());
    rest = rest.slice(cut + 1).trim();
  }
  if (rest) parts.push(rest);
  return parts;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
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

function markdownToSimpleHtml(md: string): string {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const out: string[] = [];
  let inUl = false;
  let inOl = false;
  let inCode = false;
  let codeBuf: string[] = [];
  let inTable = false;
  let tableRows: string[][] = [];

  const flushLists = () => {
    if (inUl) { out.push("</ul>"); inUl = false; }
    if (inOl) { out.push("</ol>"); inOl = false; }
  };

  const flushTable = () => {
    if (!inTable) return;
    if (tableRows.length) {
      out.push("<table>");
      tableRows.forEach((row, ri) => {
        if (row.every((c) => /^[-:\s]+$/.test(c))) return;
        const tag = ri === 0 ? "th" : "td";
        out.push(
          "<tr>" +
            row.map((c) => `<${tag}>${inlineFormat(c.trim())}</${tag}>`).join("") +
            "</tr>"
        );
      });
      out.push("</table>");
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
      if (!inUl) { flushLists(); out.push("<ul>"); inUl = true; }
      out.push(`<li>${inlineFormat(line.replace(/^\s*[-*]\s+/, ""))}</li>`);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      if (!inOl) { flushLists(); out.push("<ol>"); inOl = true; }
      out.push(`<li>${inlineFormat(line.replace(/^\s*\d+\.\s+/, ""))}</li>`);
      continue;
    }

    flushLists();

    if (/^###\s+/.test(line)) out.push(`<h3>${inlineFormat(line.replace(/^###\s+/, ""))}</h3>`);
    else if (/^##\s+/.test(line)) out.push(`<h2>${inlineFormat(line.replace(/^##\s+/, ""))}</h2>`);
    else if (/^#\s+/.test(line)) out.push(`<h1>${inlineFormat(line.replace(/^#\s+/, ""))}</h1>`);
    else if (line.trim() === "") out.push("<br/>");
    else out.push(`<p>${inlineFormat(line)}</p>`);
  }

  flushLists();
  flushTable();
  if (inCode) out.push(`<pre><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`);
  return out.join("\n");
}