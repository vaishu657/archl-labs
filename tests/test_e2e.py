"""End-to-end testing suite for AskYourDoc RAG API and /ask endpoint."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.testclient import TestClient
from src.api.app import create_app


def create_sample_pdf_bytes() -> bytes:
    """Create a minimal valid 2-page PDF with structured text."""
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
        "4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
        "5 0 obj << /Length 200 >> stream\n"
        "BT\n/F1 12 Tf\n50 700 Td\n(AskYourDoc is a modern modular RAG platform. The project architecture follows strict SOLID and OOP standards with hot-swappable interfaces for vector databases, LLMs, embeddings, and caches.) Tj\nET\n"
        "endstream\nendobj\n"
        "6 0 obj << /Length 180 >> stream\n"
        "BT\n/F1 12 Tf\n50 700 Td\n(The warranty period for all enterprise deployments is three years from installation. Technical support is available twenty-four hours a day.) Tj\nET\n"
        "endstream\nendobj\n"
        "7 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "xref\n0 8\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000240 00000 n \n0000000365 00000 n \n0000000615 00000 n \n0000000845 00000 n \ntrailer << /Size 8 /Root 1 0 R >>\nstartxref\n920\n%%EOF"
    )
    return pdf_content.encode("latin-1")


def test_e2e_flow():
    """Execute end-to-end test verifying all endpoints and features."""
    app = create_app()
    with TestClient(app) as client:
        # [1] Health check
        res = client.get("/api/health")
        assert res.status_code == 200
        health = res.json()
        assert health["status"] == "healthy"
        print("  -> [1/7] Health check passed:", health)

        # [2] POST /ask (assignment requirement)
        ask_res = client.post("/ask", json={"question": "What is Retrieval-Augmented Generation?"})
        assert ask_res.status_code == 200
        ask_data = ask_res.json()
        assert "answer" in ask_data and len(ask_data["answer"]) > 0
        assert "sources" in ask_data and len(ask_data["sources"]) > 0
        assert "page_number" in ask_data["sources"][0]
        print(f"  -> [2/7] POST /ask passed:\n     Answer: {ask_data['answer'][:120]}...\n     Sources: {len(ask_data['sources'])}")

        # [3] Document upload
        pdf_data = create_sample_pdf_bytes()
        files = {"file": ("enterprise_agreement.pdf", pdf_data, "application/pdf")}
        res_upload = client.post("/api/documents/upload", files=files)
        assert res_upload.status_code == 200
        upload_data = res_upload.json()
        doc_id = upload_data["doc_id"]
        print(f"  -> [3/7] Upload passed: doc_id={doc_id}, pages={upload_data['total_pages']}, chunks={upload_data['total_chunks']}")

        # [4] Document list
        res_list = client.get("/api/documents")
        assert res_list.status_code == 200
        assert len(res_list.json()) >= 1
        print(f"  -> [4/7] Document list passed ({len(res_list.json())} documents)")

        # [5] Synchronous query with document filter
        query_payload = {"question": "What is the warranty period for enterprise deployments?", "doc_id": doc_id}
        res_q = client.post("/api/query", json=query_payload)
        assert res_q.status_code == 200
        q_data = res_q.json()
        assert len(q_data["citations"]) > 0
        print(f"  -> [5/7] Query passed: is_cached={q_data['is_cached']}, citations={len(q_data['citations'])}")

        # [6] Cache hit verification
        res_q2 = client.post("/api/query", json=query_payload)
        assert res_q2.status_code == 200
        assert res_q2.json()["is_cached"] is True
        print(f"  -> [6/7] Cache hit verified: is_cached={res_q2.json()['is_cached']}")

        # [7] Delete uploaded document
        del_res = client.delete(f"/api/documents/{doc_id}")
        assert del_res.status_code == 200
        print(f"  -> [7/7] Document deletion passed: {del_res.json()}")

        print("\nALL END-TO-END TESTS PASSED SUCCESSFULLY! [SUCCESS]")


if __name__ == "__main__":
    test_e2e_flow()
