"""Fixtures partagées par les tests du TD06 (ne pas modifier)."""
import pytest

from slrresearch.agentic_models import AgentState, Concept, Focus, Strategy
from slrresearch.rag_models import VerifiedEvidence

COST = Concept(label="cout", terms=["cost", "dollars"])
VOTE = Concept(label="vote_majoritaire", terms=["majority voting"])
READ = Concept(label="charge_de_lecture", terms=["had to read", "reading effort"])

SEMANTIC_Q = Strategy(retriever="semantique", query_kind="question", k=3)
LEXICAL_R = Strategy(retriever="lexical", query_kind="reformulee", k=3)
HYBRID_R = Strategy(retriever="hybride", query_kind="reformulee", k=3)
HYBRID_R6 = Strategy(retriever="hybride", query_kind="reformulee", k=6)


def passage(chunk_id: str, text: str, page: int = 3, paper_id: str = "P1", score: float = 0.5) -> dict:
    """Un passage au format de sortie des retrievers du TD05."""
    return {"paper_id": paper_id, "chunk_id": chunk_id, "page": page, "text": text, "retrieval_score": score}


VOTE_PASSAGE = passage("P2_p3_c2", "Majority voting multiplied the cost by five, reaching 4.10 dollars.",
                       paper_id="P2")
READ_PASSAGE = passage("P1_p3_c2", "Screening reduced the number of abstracts that reviewers had to read by 52 percent.")
OTHER_PASSAGE = passage("P1_p5_c1", "Related work. Several position papers discuss LLM screening.", page=5)


def evidence(status: str = "verified", paper_id: str = "P2", page: int = 3,
             chunk_id: str = "P2_p3_c2") -> VerifiedEvidence:
    return VerifiedEvidence(answer="4,10 dollars", supporting_quote="Majority voting multiplied the cost by five",
                            paper_id=paper_id, chunk_id=chunk_id, page=page, status=status,
                            verification_reason="test")


def make_state(*focuses: Focus, **changes) -> AgentState:
    focuses = focuses or (Focus(question="Combien coûte le vote majoritaire ?", concepts=[COST, VOTE]),)
    return AgentState(question=focuses[0].question, focuses=list(focuses), **changes)


@pytest.fixture
def state():
    return make_state()
