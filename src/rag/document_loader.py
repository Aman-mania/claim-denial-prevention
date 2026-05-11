"""Policy document loading for Week 6 RAG.

Supported local policy source formats:
- .txt
- .md / .markdown
- .pdf via pypdf

The loader does not perform retrieval or embedding. It only converts source files
into normalized PolicyDocument objects so the downstream chunker can preserve
metadata consistently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

from src.observability import ClaimDenialError, ErrorCode, ErrorTracker
from src.rag.schemas import PolicyDocument

logger = structlog.get_logger(__name__)

SUPPORTED_POLICY_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


def _document_id(path: Path) -> str:
    stable = str(path.name).encode("utf-8")
    return hashlib.sha256(stable).hexdigest()[:16]


class PolicyDocumentLoader:
    """Load policy files from a local directory."""

    def __init__(self, *, error_tracker: ErrorTracker | None = None) -> None:
        self.error_tracker = error_tracker or ErrorTracker()

    def discover(self, raw_dir: Path) -> list[Path]:
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise ClaimDenialError(
                ErrorCode.RAG_POLICY_DOCUMENT_MISSING,
                f"Policy directory not found: {raw_dir}",
                component="rag",
                metadata={"stage": "discover_policy_documents", "path": str(raw_dir)},
            )
        return sorted(
            path for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_POLICY_SUFFIXES
        )

    def load_documents(self, raw_dir: Path) -> list[PolicyDocument]:
        """Load all supported policy files under raw_dir."""
        paths = self.discover(raw_dir)
        if not paths:
            raise ClaimDenialError(
                ErrorCode.RAG_POLICY_DOCUMENT_MISSING,
                f"No supported policy documents found in {raw_dir}.",
                component="rag",
                metadata={"stage": "load_policy_documents", "path": str(raw_dir)},
            )

        docs: list[PolicyDocument] = []
        for path in paths:
            try:
                docs.extend(self.load_file(path))
            except Exception as exc:
                self.error_tracker.record_exception(
                    exc,
                    component="rag",
                    stage="load_policy_file",
                    fallback_code=ErrorCode.RAG_DOCUMENT_PARSE_FAILED,
                    metadata={"stage": "load_policy_file", "path": str(path)},
                )
                logger.warning("policy_file_skipped", path=str(path), error=str(exc))

        if not docs:
            raise ClaimDenialError(
                ErrorCode.RAG_DOCUMENT_PARSE_FAILED,
                "Policy files were found, but none could be parsed successfully.",
                component="rag",
                metadata={"stage": "load_policy_documents", "path": str(raw_dir)},
            )

        logger.info("policy_documents_loaded", files=len(paths), documents=len(docs))
        return docs

    def load_file(self, path: Path) -> list[PolicyDocument]:
        path = Path(path)
        suffix = path.suffix.lower()
        doc_id = _document_id(path)

        if suffix in {".txt", ".md", ".markdown"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [
                PolicyDocument(
                    document_id=doc_id,
                    source_name=path.name,
                    source_type=suffix.lstrip("."),
                    source_path=str(path),
                    text=text,
                    page_number=None,
                )
            ]

        if suffix == ".pdf":
            return self._load_pdf(path=path, document_id=doc_id)

        raise ValueError(f"Unsupported policy document type: {path.suffix}")

    def _load_pdf(self, *, path: Path, document_id: str) -> list[PolicyDocument]:
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise ClaimDenialError(
                ErrorCode.RAG_DOCUMENT_PARSE_FAILED,
                "pypdf is required to parse PDF policy documents. Install pypdf or use TXT/MD policies.",
                component="rag",
                metadata={"stage": "parse_pdf", "path": str(path)},
            ) from exc

        try:
            reader = PdfReader(str(path))
            docs: list[PolicyDocument] = []
            for idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        PolicyDocument(
                            document_id=document_id,
                            source_name=path.name,
                            source_type="pdf",
                            source_path=str(path),
                            text=text,
                            page_number=idx,
                        )
                    )
            return docs
        except Exception as exc:
            raise ClaimDenialError(
                ErrorCode.RAG_DOCUMENT_PARSE_FAILED,
                f"Could not parse PDF policy document: {path}",
                component="rag",
                metadata={"stage": "parse_pdf", "path": str(path)},
            ) from exc
