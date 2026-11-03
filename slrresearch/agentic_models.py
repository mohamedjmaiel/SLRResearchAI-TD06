"""Contrats de données du TD06 - RAG agentique (code fourni, à lire).

Concept      : un concept requis par la question (libellé + termes tels qu'écrits dans le corpus)
Strategy     : une façon de chercher = (retriever, type de requête, k)
Assessment   : l'OBSERVATION après une recherche : les preuves suffisent-elles ?
Focus        : une (sous-)question suivie par l'agent, avec ses passages et sa preuve
AgentAction  : ce qu'un décideur (règles ou LLM) peut PROPOSER, rien d'autre
TraceStep    : une étape de la trace (action, statut, observation)
AgentState   : la mémoire de travail (court terme) de l'agent pendant UNE question
Episode      : un souvenir (long terme) : « pour ces concepts, cette stratégie a produit une preuve vérifiée »
AgentMemory  : la mémoire épisodique, persistée entre les questions (artifacts/agent_memory.json)
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slrresearch.rag_models import VerifiedEvidence

RetrieverName = Literal["lexical", "bm25", "semantique", "hybride"]
QueryKind = Literal["question", "reformulee"]
AssessmentStatus = Literal["suffisante", "partielle", "insuffisante"]
StepStatus = Literal["ok", "refusee", "invalide"]
Outcome = Literal["en_cours", "repondu", "partiel", "sans_preuve", "abstention", "budget_epuise"]


class Concept(BaseModel):
    """Un concept que les preuves doivent couvrir."""

    label: str = Field(min_length=1)
    terms: list[str] = Field(min_length=1)       # termes du corpus (anglais), synonymes compris


class Strategy(BaseModel):
    """Une façon de chercher. Deux stratégies égales = même retriever, même requête, même k."""

    model_config = ConfigDict(frozen=True)

    retriever: RetrieverName
    query_kind: QueryKind                        # question brute ou requête reformulée
    k: int = Field(ge=1, le=10)

    def label(self) -> str:
        return f"{self.retriever}/{self.query_kind}/k={self.k}"


class Assessment(BaseModel):
    """Observation : le meilleur passage couvre-t-il tous les concepts requis ?"""

    status: AssessmentStatus
    best_chunk_id: str | None = None
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class Focus(BaseModel):
    """Une (sous-)question : ce qui a été essayé, trouvé, observé et prouvé."""

    question: str
    concepts: list[Concept] = Field(min_length=1)
    hint: Strategy | None = None                 # stratégie suggérée par la mémoire épisodique
    tried: list[Strategy] = Field(default_factory=list)
    passages: list[dict] = Field(default_factory=list)      # dédupliqués, dans l'ordre de découverte
    assessment: Assessment | None = None
    winning: Strategy | None = None              # stratégie qui a rendu les preuves suffisantes
    evidence: VerifiedEvidence | None = None
    context_chunk_ids: list[str] = Field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        return self.assessment is not None and self.assessment.status == "suffisante"

    def labels(self) -> list[str]:
        return [concept.label for concept in self.concepts]


class AgentAction(BaseModel):
    """Contrat d'action. Les limites (retriever autorisé, k maximal, budget) sont
    contrôlées par le logiciel (check_action), pas par ce schéma : une proposition
    peut être bien formée et pourtant refusée."""

    action: Literal["search", "decompose", "answer", "abstain"]
    focus: int = Field(default=0, ge=0)
    retriever: str | None = None
    query_kind: QueryKind | None = None
    k: int | None = None
    reason: str = ""

    @model_validator(mode="after")
    def check_fields(self):
        search_fields = (self.retriever, self.query_kind, self.k)
        if self.action == "search" and any(value is None for value in search_fields):
            raise ValueError("search exige retriever, query_kind et k")
        if self.action != "search" and any(value is not None for value in search_fields):
            raise ValueError(f"{self.action} n'utilise ni retriever, ni query_kind, ni k")
        return self

    def strategy(self) -> Strategy:
        return Strategy(retriever=self.retriever, query_kind=self.query_kind, k=self.k)


class TraceStep(BaseModel):
    step: int
    action: dict | None
    status: StepStatus
    observation: str


class AgentState(BaseModel):
    """Mémoire de travail : tout ce que l'agent sait avant sa prochaine décision."""

    question: str
    config: str = "agentique"                    # naive, advanced, agentique, agentique+memoire...
    focuses: list[Focus] = Field(min_length=1)
    decomposed: bool = False
    steps: int = 0
    llm_calls: int = 0
    retriever_calls: int = 0
    trace: list[TraceStep] = Field(default_factory=list)
    outcome: Outcome = "en_cours"
    duration_ms: float = 0.0

    def record(self, action: AgentAction | None, status: StepStatus, observation: str) -> None:
        """Une étape consommée (y compris une action refusée ou invalide) : trace + compteur."""
        self.steps += 1
        self.trace.append(TraceStep(step=self.steps, action=action.model_dump() if action else None,
                                    status=status, observation=observation))


class Episode(BaseModel):
    """Souvenir : concepts de la (sous-)question -> stratégie gagnante (preuve vérifiée)."""

    question: str
    concepts: list[str] = Field(min_length=1)    # libellés triés
    strategy: Strategy
    uses: int = Field(default=1, ge=1)


class AgentMemory(BaseModel):
    episodes: list[Episode] = Field(default_factory=list)


class EvalQuestion(BaseModel):
    """Question annotée du jeu d'évaluation (data/eval_td06.json)."""

    id: str
    question: str
    kind: str                                    # facile, vocabulaire, multi_aspects, hors_corpus...
    expected: Literal["repondre", "abstention"]
    acceptable_pages: list[tuple[str, int]] = Field(default_factory=list)
    aspects: int = Field(default=1, ge=1)        # pages DISTINCTES à citer pour une réponse complète
