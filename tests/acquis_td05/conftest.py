"""Fixtures partagées par les tests du TD05 (non-régression, ne pas modifier)."""
import pytest

from slrresearch.rag_models import Chunk


@pytest.fixture
def chunks():
    return [
        Chunk(paper_id="P1", chunk_id="P1_p1_c1", page=1,
              text="Systematic reviews require reviewers to read many abstracts."),
        Chunk(paper_id="P1", chunk_id="P1_p2_c1", page=2,
              text="The best configuration reached a recall of 0.95 on four reviews."),
        Chunk(paper_id="P2", chunk_id="P2_p3_c1", page=3,
              text="Sensitivity was 1.00 with few-shot chain of thought prompting."),
    ]


@pytest.fixture
def selected(chunks):
    """Passages transmis au LLM (format de sortie des retrievers)."""
    return [{**chunk.model_dump(), "retrieval_score": score}
            for chunk, score in zip(chunks, (0.2, 0.9, 0.5))]
