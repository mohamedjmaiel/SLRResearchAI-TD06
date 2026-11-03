"""Données et artefacts du TD06 (code fourni, à lire).

Entrées   data/eval_td06.json             jeu d'évaluation annoté (pages acceptables, aspects)
          artifacts/ (TD03, TD04, TD05)   même corpus que le TD05 (load_rag_handoff)
Sorties   artifacts/agent_memory.json     mémoire épisodique (persiste d'une exécution à l'autre)
          artifacts/rag_comparison.json   comparaison expérimentale naïf / avancé / agentique
          artifacts/verified_evidence.json preuves par (sous-)question, pour le WritingAgent
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from slrresearch.agentic_models import AgentMemory, AgentState, EvalQuestion
from slrresearch.rag_artifacts import ARTIFACTS, EVIDENCE_PATH, ROOT

EVAL_PATH = ROOT / "data" / "eval_td06.json"
MEMORY_PATH = ARTIFACTS / "agent_memory.json"
REPORT_PATH = ARTIFACTS / "rag_comparison.json"


def load_eval_set(path: Path = EVAL_PATH) -> list[EvalQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion.model_validate(item) for item in data["questions"]]


def annotation_for(question: str, path: Path = EVAL_PATH) -> EvalQuestion | None:
    wanted = " ".join(question.casefold().split())
    return next((item for item in load_eval_set(path)
                 if " ".join(item.question.casefold().split()) == wanted), None)


def load_memory(path: Path = MEMORY_PATH) -> AgentMemory:
    """Mémoire vide si le fichier n'existe pas encore."""
    if not path.exists():
        return AgentMemory()
    return AgentMemory.model_validate_json(path.read_text(encoding="utf-8"))


def save_memory(memory: AgentMemory, path: Path = MEMORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory.model_dump_json(indent=1), encoding="utf-8")


def save_report(rows: list[dict], summary: list[dict], path: Path = REPORT_PATH) -> None:
    payload = {"schema_version": "1.0", "origin": "RAG_TD06",
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "protocol": "même corpus, mêmes questions, même contrôle des preuves ; "
                           "configurations : naïve, avancée, agentique, agentique + mémoire",
               "summary": summary, "runs": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def save_agentic_evidence(state: AgentState, topic: str, path: Path = EVIDENCE_PATH) -> list[dict]:
    """Une preuve par (sous-)question documentée ; seules les « verified » sont autorisées à la rédaction."""
    payload = {"schema_version": "1.0", "origin": "RAG_TD06", "topic": topic,
               "source_artifacts": ["search_results.json", "screening_decisions.json",
                                    "document_manifest.json"], "evidence": []}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("topic") == topic:
                payload["evidence"] = [e for e in previous.get("evidence", [])
                                       if e.get("question") != state.question]
        except json.JSONDecodeError:
            pass
    records = []
    for focus in state.focuses:
        if focus.evidence is None:
            continue
        records.append({
            "question": state.question,
            "focus_question": focus.question,
            "config": state.config,
            "strategy": focus.winning.label() if focus.winning else None,
            "retrieved_chunk_ids": focus.context_chunk_ids,
            **focus.evidence.model_dump(),
            "authorized_for_writing": focus.evidence.status == "verified",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    payload["evidence"].extend(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return records
