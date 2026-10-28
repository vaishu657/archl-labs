"""FastAPI route definitions for AskMyPDF / AskYourDoc."""

import json
import logging
from pathlib import Path
from typing import List
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.api.schemas import (
    AskRequest,
    AskResponse,
    CacheStatsResponse,
    CitationResponse,
    DocumentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
    UploadResponse,
)
from src.services.container import container

logger = logging.getLogger("AskMyPDF.Routes")
router = APIRouter()
templates = Jinja2Templates(directory="templates")


def ensure_default_document_indexed() -> None:
    """Ensure the default document (Document (4).pdf / sample_document.pdf) is indexed."""
    docs = container.document_service.list_documents()
    if docs:
        return

    candidate_paths = [
        container.settings.BASE_DIR / "sample_document.pdf",
        container.settings.BASE_DIR / "data" / "sample_document.pdf",
        container.settings.BASE_DIR / "data" / "documents" / "Document (4).pdf",
        Path(r"c:\Users\manor\Downloads\Document (4).pdf")
    ]

    for p in candidate_paths:
        if p.exists():
            try:
                pdf_bytes = p.read_bytes()
                container.document_service.ingest_pdf("Document (4).pdf", pdf_bytes)
                logger.info("Auto-indexed default PDF document from '%s'", p)
                return
            except Exception as e:
                logger.warning("Could not auto-index from '%s': %s", p, e)


@router.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Serve the main interactive single-page application."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": container.settings.APP_NAME,
            "llm_provider": container.llm.provider_name,
            "embedding_model": container.settings.EMBEDDING_MODEL_NAME,
        }
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    POST /ask endpoint matching the Archle Labs assignment requirements.

    Example request:
    {
      "question": "What is Retrieval-Augmented Generation?"
    }

    Example response:
    {
      "answer": "Retrieval-Augmented Generation is a technique that combines information retrieval with text generation...",
      "sources": [
        {
          "page_number": 1,
          "chunk_id": "...",
          "filename": "Document (4).pdf",
          "excerpt": "..."
        }
      ]
    }
    """
    ensure_default_document_indexed()

    try:
        result = container.workflow.run(
            query=request.question,
            filter_doc_id=request.doc_id
        )

        sources = [
            SourceReference(
                page_number=c.page_number,
                chunk_id=c.chunk_id,
                filename=c.filename,
                excerpt=c.excerpt,
                similarity_score=c.similarity_score
            )
            for c in result.citations
        ]

        return AskResponse(
            answer=result.answer,
            sources=sources
        )
    except Exception as e:
        logger.error("Query execution failed for '%s': %s", request.question, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}"
        )


@router.post("/api/documents/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only valid .pdf files are accepted."
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is empty."
            )

        meta = container.document_service.ingest_pdf(filename=file.filename, file_bytes=content)

        return UploadResponse(
            message=f"Successfully indexed '{file.filename}'",
            doc_id=meta.doc_id,
            filename=meta.filename,
            total_pages=meta.total_pages,
            total_chunks=meta.total_chunks
        )
    except ValueError as e:
        logger.warning("Upload validation failed: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to process uploaded PDF '%s': %s", file.filename, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while indexing the document: {str(e)}"
        )


@router.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents():
    """Retrieve all indexed documents."""
    docs = container.document_service.list_documents()
    return [DocumentResponse(**d) for d in docs]


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document and all its indexed chunks from the system."""
    success = container.document_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Document '{doc_id}' deleted successfully", "doc_id": doc_id}


@router.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Execute synchronous document question answering for the interactive web UI."""
    ensure_default_document_indexed()

    try:
        result = container.workflow.run(
            query=request.question,
            filter_doc_id=request.doc_id
        )

        citations_res = [
            CitationResponse(
                doc_id=c.doc_id,
                filename=c.filename,
                page_number=c.page_number,
                chunk_id=c.chunk_id,
                excerpt=c.excerpt,
                similarity_score=c.similarity_score
            )
            for c in result.citations
        ]

        return QueryResponse(
            question=result.question,
            answer=result.answer,
            citations=citations_res,
            is_cached=result.is_cached,
            execution_time_ms=result.execution_time_ms,
            llm_provider=result.llm_provider
        )
    except Exception as e:
        logger.error("Query execution failed for '%s': %s", request.question, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}"
        )


@router.post("/api/query/stream")
async def query_documents_stream(request: QueryRequest):
    """Server-Sent Events (SSE) streaming endpoint for real-time token emission."""
    ensure_default_document_indexed()

    def event_generator():
        try:
            for event in container.workflow.stream(query=request.question, filter_doc_id=request.doc_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error("Stream generation error for '%s': %s", request.question, e, exc_info=True)
            err_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/api/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Retrieve cache hit/miss statistics and storage metrics."""
    stats = container.cache.get_stats()
    return CacheStatsResponse(
        total_entries=stats.get("total_entries", 0),
        namespaces=stats.get("namespaces", {}),
        file_size_kb=stats.get("file_size_kb", 0.0),
        hits=stats.get("hits", 0),
        misses=stats.get("misses", 0),
        hit_rate_percentage=stats.get("hit_rate_percentage", 0.0)
    )


@router.post("/api/cache/clear")
async def clear_cache():
    """Flush the cache."""
    container.cache.clear()
    return {"message": "Cache successfully cleared."}


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """System health and component diagnostics."""
    docs = container.document_service.list_documents()
    return HealthResponse(
        status="healthy",
        app_name=container.settings.APP_NAME,
        llm_provider=container.llm.provider_name,
        vector_store=type(container.vector_store).__name__,
        total_documents=len(docs)
    )
