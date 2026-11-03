"""Non-régression du TD05 : transmission TD03 → TD04 → documents → RAG → artefact."""
import json
import shutil

import pytest

from slrresearch.rag_artifacts import (ARTIFACTS, load_rag_handoff, relevant_pages_for,
                                       save_evidence_artifact)
from slrresearch.rag_models import ProposedEvidence, RagRun, VerifiedEvidence

W1001 = "https://openalex.org/W1001"
S2 = "s2-7f3a"


def test_td05_base_handoff_ne_garde_que_les_papiers_inclus_et_disponibles():
    documents, lineage = load_rag_handoff()
    assert [d.paper_id for d in documents] == [W1001, S2]
    assert (lineage["found"], lineage["included"], lineage["documents_available"],
            lineage["rag_documents"]) == (4, 3, 2, 2)
    assert lineage["excluded_from_rag"]["crossref:10.5555/SLR.2022.207"] == "exclu au TD04"
    assert "indisponible" in lineage["excluded_from_rag"]["https://openalex.org/W1002"]


def test_td05_base_handoff_refuse_une_decision_orpheline(tmp_path):
    for name in ("search_results.json", "screening_decisions.json",
                 "document_manifest.json", "extracted_pages.json"):
        shutil.copy(ARTIFACTS / name, tmp_path / name)
    screening = json.loads((tmp_path / "screening_decisions.json").read_text(encoding="utf-8"))
    screening["decisions"].append({"paper_id": "W9999", "final_decision": "include",
                                   "decider": "humain", "reason": "article inconnu"})
    (tmp_path / "screening_decisions.json").write_text(json.dumps(screening), encoding="utf-8")
    with pytest.raises(ValueError, match="absents du TD03"):
        load_rag_handoff(tmp_path)


def test_td05_base_annotations_par_article_et_page():
    question = "Comment les LLM améliorent-ils le rappel du screening des SLR ?"
    assert relevant_pages_for(question) == {(W1001, 3), (S2, 3)}
    assert relevant_pages_for("Une question jamais annotée ?") is None


def test_td05_base_preuve_devient_un_artefact_pour_le_td_suivant(tmp_path):
    target = tmp_path / "verified_evidence.json"
    proposal = ProposedEvidence(answer="Rappel de 0,95.", supporting_quote="reached a recall of 0.95",
                                paper_id="P1", chunk_id="P1_p2_c1", page=2)
    run = RagRun(question="Q1 ?", retrieval_query="recall",
                 selected_chunks=[{"chunk_id": "P1_p2_c1"}],
                 evidence=VerifiedEvidence(**proposal.model_dump(), status="human_review",
                                           verification_reason="citation approximative"))
    save_evidence_artifact(run, topic="sujet", retriever="lexical", path=target)
    run.evidence = run.evidence.model_copy(update={"status": "verified"})
    save_evidence_artifact(run, topic="sujet", retriever="lexical", path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["origin"] == "RAG_TD05" and len(payload["evidence"]) == 1
    assert payload["evidence"][0]["authorized_for_writing"] is True
    with pytest.raises(ValueError):
        save_evidence_artifact(RagRun(question="Q", retrieval_query="q", abstained=True),
                               topic="sujet", retriever="lexical", path=target)
