import sqlite3
from datetime import datetime
from pathlib import Path

from app.documents.models import DocumentRecord, DocumentStatus


class DocumentRepository:
    """SQLite-backed repository for document metadata."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        """Create the document table if it does not exist."""

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    file_size_bytes INTEGER NOT NULL,
                    page_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            connection.commit()

    def add(self, document: DocumentRecord) -> None:
        """Persist a new document record."""

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    filename,
                    source_path,
                    file_hash,
                    file_size_bytes,
                    page_count,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.filename,
                    document.source_path,
                    document.file_hash,
                    document.file_size_bytes,
                    document.page_count,
                    document.status.value,
                    document.created_at.isoformat(),
                ),
            )

            connection.commit()

    def get_by_hash(
        self,
        file_hash: str,
    ) -> DocumentRecord | None:
        """Return a document by SHA-256 hash."""

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    source_path,
                    file_hash,
                    file_size_bytes,
                    page_count,
                    status,
                    created_at
                FROM documents
                WHERE file_hash = ?
                """,
                (file_hash,),
            ).fetchone()

        if row is None:
            return None

        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            source_path=row["source_path"],
            file_hash=row["file_hash"],
            file_size_bytes=row["file_size_bytes"],
            page_count=row["page_count"],
            status=DocumentStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )


    def list_all(self) -> list[DocumentRecord]:
        """Return every document, most recently created first."""

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    source_path,
                    file_hash,
                    file_size_bytes,
                    page_count,
                    status,
                    created_at
                FROM documents
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [
            DocumentRecord(
                document_id=row["document_id"],
                filename=row["filename"],
                source_path=row["source_path"],
                file_hash=row["file_hash"],
                file_size_bytes=row["file_size_bytes"],
                page_count=row["page_count"],
                status=DocumentStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def get_by_id(
        self,
        document_id: str,
    ) -> DocumentRecord | None:
        """Return a document by document ID."""

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                """
                SELECT
                    document_id,
                    filename,
                    source_path,
                    file_hash,
                    file_size_bytes,
                    page_count,
                    status,
                    created_at
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            source_path=row["source_path"],
            file_hash=row["file_hash"],
            file_size_bytes=row["file_size_bytes"],
            page_count=row["page_count"],
            status=DocumentStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )