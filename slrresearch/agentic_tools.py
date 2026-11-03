"""Outils, vocabulaire et garde-fous du TD06 - RAG agentique (code fourni, à lire).

Vocabulaire contrôlé   GLOSSARY : concepts discriminants de la revue (français -> termes du corpus)
Stratégies             STRATEGIES : l'« échelle » des façons de chercher, de la moins chère à la plus large
Budgets                MAX_STEPS, MAX_RETRIEVALS, MAX_K : l'agent ne peut pas boucler sans fin
Boîte à outils         Toolbox : retrievers (TD05), extracteur (TD05), décomposeur (LLM), mémoire
Décideur LLM           make_llm_decider : le LLM PROPOSE une action JSON ; le logiciel dispose
"""
import json
import re
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field

from slrresearch.agentic_models import AgentAction, AgentMemory, AgentState, Concept, Strategy
from slrresearch.rag_pipeline import DEFAULT_K, Extractor, Retriever
from slrresearch.search_agent import recover_rejected_generation
from slrresearch.search_plan import call_llm, extract_json_object

MAX_STEPS = 10           # étapes de la boucle (une action refusée consomme aussi une étape)
MAX_RETRIEVALS = 8       # appels de retriever pour une question
MAX_K = 6                # au-delà, le contexte transmis au LLM serait trop long et dilué
MEMORY_THRESHOLD = 0.5   # similarité de Jaccard minimale pour réutiliser un souvenir

# Échelle des stratégies : on commence comme le RAG naïf (TD05), puis on corrige.
STRATEGIES: tuple[Strategy, ...] = (
    Strategy(retriever="semantique", query_kind="question", k=3),     # = RAG naïf du TD05
    Strategy(retriever="lexical", query_kind="reformulee", k=3),      # reformuler (vocabulaire du corpus)
    Strategy(retriever="hybride", query_kind="reformulee", k=3),      # changer de retriever
    Strategy(retriever="hybride", query_kind="reformulee", k=6),      # élargir le Top-k
)

# Vocabulaire contrôlé de la revue. Seuls les concepts DISCRIMINANTS y figurent :
# « LLM » ou « screening » apparaissent dans presque tous les passages et ne
# permettent pas de juger si un passage répond à la question.
GLOSSARY: dict[str, dict[str, list[str]]] = {
    "rappel": {"fr": ["rappel", "sensibilité"], "en": ["recall", "sensitivity"]},
    "precision": {"fr": ["précision", "spécificité", "faux positifs"],
                  "en": ["precision", "specificity", "false positives"]},
    "few_shot": {"fr": ["few-shot", "quelques exemples"], "en": ["few-shot"]},
    "chaine_de_pensee": {"fr": ["chaîne de pensée", "chaîne de raisonnement"],
                         "en": ["chain of thought"]},
    "charge_de_lecture": {"fr": ["charge de lecture", "effort de lecture", "résumés à lire"],
                          "en": ["had to read", "reading effort"]},
    "cout": {"fr": ["coût", "coûts", "coûte", "coûtent", "surcoût", "prix"],
             "en": ["cost", "dollars"]},
    "vote_majoritaire": {"fr": ["vote majoritaire", "vote à la majorité"],
                         "en": ["majority voting"]},
    "supervision_humaine": {"fr": ["contrôle humain", "supervision humaine", "humain dans la boucle",
                                   "intervention humaine"],
                            "en": ["human oversight", "human-in-the-loop", "escalated to a human"]},
    "decision_finale": {"fr": ["décision finale"], "en": ["final inclusion decision", "final decision"]},
    "accord_inter_evaluateurs": {"fr": ["accord inter-évaluateurs", "accord entre relecteurs", "kappa"],
                                 "en": ["inter-rater agreement", "agreement", "kappa"]},
    "reproductibilite": {"fr": ["reproductibilité", "reproductible"],
                         "en": ["reproducibility", "reproducible"]},
    "fuite_de_donnees": {"fr": ["fuite de données", "contamination"], "en": ["data leakage", "leakage"]},
}


class ActionRefused(Exception):
    """Une proposition refusée par le logiciel devient une observation (statut refusee/invalide)."""

    def __init__(self, message: str, status: str = "refusee"):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Correspondance de termes et concepts requis
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Minuscules, apostrophes unifiées, espaces simples."""
    return " ".join(text.replace("’", "'").casefold().split())


def contains_term(text: str, term: str) -> bool:
    """Le terme (mot ou expression) apparaît-il comme unité entière dans le texte ?

    contains_term("high recall", "recall") -> True
    contains_term("a disagreement", "agreement") -> False   (pas un mot entier)
    """
    pattern = r"(?<![\w@-])" + re.escape(normalize(term)) + r"(?![\w@-])"
    return re.search(pattern, normalize(text)) is not None


def concepts_from_question(question: str, glossary: dict = GLOSSARY) -> list[Concept]:
    """Concepts du vocabulaire contrôlé mentionnés par la question (en français ou en anglais)."""
    found = []
    for label, entry in glossary.items():
        if any(contains_term(question, term) for term in entry["fr"] + entry["en"]):
            found.append(Concept(label=label, terms=entry["en"]))
    return found


def concepts_from_td02(validated, others=()) -> list[Concept]:
    """Pont TD02 -> TD06 : les concepts validés par l'humain deviennent les concepts requis.

    Le corpus a déjà été filtré sur le SUJET (TD03, TD04) : un concept partagé avec les
    autres RQ du sujet (ex. « screening ») ne discrimine pas les passages. On ne garde
    donc que les concepts propres à la RQ (tous, si aucun ne lui est propre).
    """
    shared = {group.label.casefold() for other in others for group in other.concepts}
    concepts = [Concept(label=group.label, terms=group.terms) for group in validated.concepts]
    specific = [concept for concept in concepts if concept.label.casefold() not in shared]
    return specific or concepts


# ---------------------------------------------------------------------------
# Contrat d'action et boîte à outils
# ---------------------------------------------------------------------------

def validate_action(proposal) -> AgentAction:
    """Applique le contrat AgentAction à une proposition (dict ou AgentAction)."""
    data = proposal.model_dump() if isinstance(proposal, AgentAction) else proposal
    try:
        return AgentAction.model_validate(data)
    except ValueError as error:
        raise ActionRefused(f"Action hors contrat : {error}", status="invalide") from error


@dataclass
class Toolbox:
    """Ce que l'agent a le droit d'utiliser (moindre privilège)."""

    retrievers: dict[str, Retriever]               # make_retrievers du TD05
    extractor: Extractor                           # build_llm_extractor du TD05
    decomposer: Callable[[str], list[str]] | None = None
    memory: AgentMemory | None = None
    answer_k: int = DEFAULT_K                      # passages transmis au LLM pour répondre


class SubQuestions(BaseModel):
    """Sortie structurée du décomposeur : 1 sous-question = question non décomposable."""

    sub_questions: list[str] = Field(min_length=1, max_length=3)


def build_decomposition_prompt(question: str) -> str:
    return f"""[DÉCOMPOSITION]
Tu prépares la recherche documentaire d'une revue systématique.
Si la question porte sur plusieurs aspects indépendants, découpe-la en 2 ou 3
sous-questions autonomes (une par aspect). Sinon, renvoie la question telle quelle.
Ne réponds pas à la question.

[QUESTION]
{question}
"""


def make_decomposer(model) -> Callable[[str], list[str]]:
    """Décomposeur LLM (sortie structurée SubQuestions)."""
    structured = model.with_structured_output(SubQuestions)

    def decompose(question: str) -> list[str]:
        result = structured.invoke(build_decomposition_prompt(question))
        if isinstance(result, dict):
            result = SubQuestions.model_validate(result)
        return [item.strip() for item in result.sub_questions if item.strip()]

    return decompose


# ---------------------------------------------------------------------------
# Décideur LLM (niveau 3, comme au TD03) : le LLM propose, le logiciel dispose
# ---------------------------------------------------------------------------

def describe_state(state: AgentState, retrievers: list[str]) -> str:
    """La partie de l'état utile à la décision (≠ état complet : pas les textes)."""
    view = {
        "question": state.question,
        "pas_utilises": state.steps,
        "pas_max": MAX_STEPS,
        "appels_retriever": state.retriever_calls,
        "decomposee": state.decomposed,
        "retrievers_autorises": sorted(retrievers),
        "k_max": MAX_K,
        "echelle_strategies": [s.model_dump() for s in STRATEGIES],
        "focus": [
            {
                "index": index,
                "question": focus.question,
                "concepts": focus.labels(),
                "preuves": focus.assessment.status if focus.assessment else "non_cherchees",
                "manque": focus.assessment.missing if focus.assessment else focus.labels(),
                "essayees": [s.model_dump() for s in focus.tried],
                "suggestion_memoire": focus.hint.model_dump() if focus.hint else None,
            }
            for index, focus in enumerate(state.focuses)
        ],
        "historique": [
            {"pas": t.step, "action": (t.action or {}).get("action"), "statut": t.status}
            for t in state.trace[-6:]
        ],
    }
    return json.dumps(view, ensure_ascii=False, indent=1)


def build_decision_prompt(state: AgentState, retrievers: list[str]) -> str:
    return f"""Tu pilotes un agent RAG qui documente une question de revue systématique.

[OBJECTIF]
Réunir, pour chaque focus, un passage qui couvre tous ses concepts ; puis répondre.
S'abstenir si aucune preuve suffisante n'existe : une réponse non prouvée est pire qu'une abstention.

[ÉTAT]
{describe_state(state, retrievers)}
[ACTIONS]
- Chercher    : {{"action": "search", "focus": <index>, "retriever": "<retriever autorisé>", "query_kind": "question" | "reformulee", "k": <1..k_max>}}
- Décomposer  : {{"action": "decompose", "focus": 0}}   (une seule fois, si les preuves sont partielles)
- Répondre    : {{"action": "answer"}}                   (au moins un focus aux preuves suffisantes)
- S'abstenir  : {{"action": "abstain"}}
[RÈGLES]
- Réponds uniquement par un objet JSON.
- Essaie d'abord la suggestion de la mémoire, puis l'échelle des stratégies, sans répéter une stratégie déjà essayée.
- Réponds quand chaque focus a des preuves suffisantes ou qu'il ne reste aucune stratégie à essayer.

[DÉCISION RAG]
"""


def parse_agent_action(text: str) -> AgentAction:
    """Sortie texte du LLM -> AgentAction, ou ActionRefused(status="invalide")."""
    try:
        return AgentAction.model_validate_json(extract_json_object(text))
    except ValueError as error:
        raise ActionRefused(f"Sortie LLM hors contrat : {error}", status="invalide") from error


def make_llm_decider(model, retrievers: list[str]) -> Callable[[AgentState], AgentAction]:
    """Adapte le LLM à l'interface decide(state). Chaque décision coûte un appel LLM."""

    def decide(state: AgentState) -> AgentAction:
        prompt = build_decision_prompt(state, retrievers)
        state.llm_calls += 1
        try:
            text = call_llm(model, prompt)
        except Exception as error:          # réponse rejetée par le fournisseur (cf. TD03)
            text = recover_rejected_generation(error)
            if text is None:
                raise
        return parse_agent_action(text)

    return decide
