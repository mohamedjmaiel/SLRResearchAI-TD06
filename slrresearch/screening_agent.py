"""ScreeningAgent du TD04 : décider, s'abstenir, escalader.

Pour chaque article (titre + résumé), deux évaluateurs indépendants jugent :
    les RÈGLES (protocole exécutable)   et   le LLM (jugement structuré, contrôlé)
puis une politique de décision choisit l'une de trois issues :
    agir (include / exclude automatique) · s'abstenir · escalader vers un humain.

Partie A  du protocole aux preuves        TODO 1 à 3
Partie B  le LLM, second évaluateur       TODO 4 et 5
Partie C  décider, s'abstenir, escalader  TODO 6 à 8
Partie D  boucler avec l'humain           TODO 9 et 10

Ce fichier est le seul à modifier : TODO 1 à TODO 10.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field

from slrresearch.abstracts import ScreeningPaper
from slrresearch.search_plan import ConceptGroup, call_llm, extract_json_object

DEFAULT_THRESHOLD = 0.75   # confiance minimale pour une décision automatique
EXCLUDE_MARGIN = 0.10      # exclure à tort coûte plus cher : on exige davantage
THRESHOLDS = (0.5, 0.6, 0.75, 0.9)

Decision = Literal["include", "exclude", "uncertain"]
Decider = Literal["regles", "llm", "accord", "humain"]


# ===========================================================================
# Contrats (code fourni, à lire)
# ===========================================================================

class Criterion(BaseModel):
    """Un critère d'éligibilité : satisfait si l'un de ses termes apparaît."""

    id: str                                   # C1, C2... (inclusion) ou E1, E2... (exclusion)
    label: str
    terms: list[str] = Field(min_length=1)


class Protocol(BaseModel):
    """Protocole de screening d'une RQ : tous les C requis, aucun E toléré."""

    question: str
    inclusion: list[Criterion] = Field(min_length=1)
    exclusion: list[Criterion] = Field(default_factory=list)


class Evidence(BaseModel):
    """Une preuve = un critère + un extrait du titre ou du résumé."""

    criterion: str
    quote: str = Field(min_length=3)


class ScreeningDecision(BaseModel):
    """Une décision explicable : quoi, avec quelle confiance, pourquoi, preuves à l'appui."""

    paper_id: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    decider: Decider


class LLMJudgment(BaseModel):
    """Ce que le LLM doit renvoyer (contrat de sortie, comme au TD02)."""

    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)


class ScreeningRecord(BaseModel):
    """Le dossier d'un article : deux avis, une issue, une décision finale éventuelle."""

    paper_id: str
    title: str
    rules: ScreeningDecision
    llm: ScreeningDecision
    route: Literal["auto", "humain"]
    escalation: str = ""                      # pourquoi l'humain est sollicité
    final: ScreeningDecision | None = None    # None : en attente d'un humain


class ScreeningState(BaseModel):
    protocol: Protocol
    threshold: float
    records: list[ScreeningRecord] = Field(default_factory=list)

    def human_queue(self) -> list[ScreeningRecord]:
        return [r for r in self.records if r.final is None]

    def counts(self) -> dict[str, int]:
        finals = [r.final.decision for r in self.records if r.final]
        return {
            "examinés": len(self.records),
            "inclus": finals.count("include"),
            "exclus": finals.count("exclude"),
            "en_attente_humain": len(self.human_queue()),
            "décidés_par_un_humain": sum(1 for r in self.records if r.final and r.final.decider == "humain"),
        }


def paper_text(paper: ScreeningPaper) -> str:
    """Code fourni : le texte examiné (titre + résumé)."""
    return paper.title + ". " + (paper.abstract or "")


def sentences(text: str) -> list[str]:
    """Code fourni : découpe en phrases (une preuve = une phrase)."""
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text) if s.strip()]


def normalize(text: str) -> str:
    """Code fourni : minuscules et espaces compactés (comparaison robuste)."""
    return " ".join(text.lower().split())


DEFAULT_EXCLUSION = [
    Criterion(id="E1", label="domaine clinique", terms=["clinical trial", "patients", "medical"]),
    Criterion(id="E2", label="publication non éligible", terms=["editorial", "erratum", "keynote"]),
]


# ===========================================================================
# Partie A - Du protocole aux preuves
# ===========================================================================

def protocol_from_concepts(
    question: str,
    concepts: list[ConceptGroup],
    exclusion: list[Criterion] | None = None,
) -> Protocol:
    """TODO 1 - Pont TD02 : chaque concept validé devient un critère d'inclusion."""
    inclusion = [
        Criterion(id=f"C{index}", label=group.label, terms=list(group.terms))
        for index, group in enumerate(concepts, start=1)
    ]
    return Protocol(
        question=question,
        inclusion=inclusion,
        exclusion=list(DEFAULT_EXCLUSION if exclusion is None else exclusion),
    )


def find_evidence(text: str, criterion: Criterion) -> Evidence | None:
    """TODO 2 - Chercher une preuve d'un critère : la première phrase qui contient un de ses termes.

    Un terme est un MOT ENTIER, sans tenir compte de la casse, pluriel en « s »
    accepté : « LLM » trouve « LLMs », mais pas « LLMOps ».
    """
    for sentence in sentences(text):
        for term in criterion.terms:
            pattern = r"\b" + re.escape(term) + r"s?\b"
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                return Evidence(criterion=criterion.id, quote=sentence)
    return None


def rule_screen(paper: ScreeningPaper, protocol: Protocol) -> ScreeningDecision:
    """TODO 3 - Premier évaluateur : les règles du protocole, preuves à l'appui."""
    if not paper.abstract:
        return ScreeningDecision(paper_id=paper.id, decision="uncertain", confidence=0.0,
                                 reasons=["résumé absent : information insuffisante"], decider="regles")
    text = paper_text(paper)
    excluded = [e for c in protocol.exclusion if (e := find_evidence(text, c))]
    found = {c.id: find_evidence(text, c) for c in protocol.inclusion}
    matched = [e for e in found.values() if e]
    missing = [cid for cid, e in found.items() if e is None]
    if excluded:
        return ScreeningDecision(
            paper_id=paper.id, decision="exclude", confidence=0.9, evidence=excluded,
            reasons=[f"critère d'exclusion {e.criterion} présent" for e in excluded], decider="regles")
    if not missing:
        return ScreeningDecision(
            paper_id=paper.id, decision="include", confidence=0.9, evidence=matched,
            reasons=["tous les critères d'inclusion sont présents"], decider="regles")
    if not matched:
        return ScreeningDecision(
            paper_id=paper.id, decision="exclude", confidence=0.85,
            reasons=["aucun critère d'inclusion présent"], decider="regles")
    return ScreeningDecision(
        paper_id=paper.id, decision="uncertain", confidence=0.5, evidence=matched,
        reasons=[f"critère absent : {cid}" for cid in missing], decider="regles")


# ===========================================================================
# Partie B - Le LLM, second évaluateur contrôlé
# ===========================================================================

def build_screening_prompt(paper: ScreeningPaper, protocol: Protocol) -> str:
    """Code fourni : prompt du LLM évaluateur."""
    def describe(criteria):
        return "\n".join(f"- {c.id} · {c.label} : {', '.join(c.terms)}" for c in criteria) or "- (aucun)"
    return f"""Tu es relecteur pour une revue systématique de la littérature (screening titre/résumé).

[QUESTION DE RECHERCHE]
{protocol.question}

[CRITÈRES D'INCLUSION] (tous requis)
{describe(protocol.inclusion)}

[CRITÈRES D'EXCLUSION] (un seul suffit)
{describe(protocol.exclusion)}

[ARTICLE]
id : {paper.doi or paper.id}
titre : {paper.title}
résumé : {paper.abstract or "(indisponible)"}

[RÈGLES]
- Réponds uniquement par un objet JSON :
  {{"decision": "include|exclude|uncertain", "confidence": 0.0-1.0,
    "reasons": ["..."], "evidence": [{{"criterion": "C1", "quote": "extrait EXACT du titre ou du résumé"}}]}}
- Chaque citation doit être recopiée mot pour mot depuis le titre ou le résumé.
- Si l'information est insuffisante, réponds "uncertain" : ne devine pas.
- N'appelle aucune fonction : écris l'objet JSON comme simple texte.

[SCREENING]
"""


def parse_judgment(text: str, paper_id: str) -> ScreeningDecision:
    """TODO 4 - Sortie du LLM -> ScreeningDecision. Hors contrat : abstention, pas plantage."""
    try:
        judgment = LLMJudgment.model_validate_json(extract_json_object(text))
    except ValueError as error:
        first = str(error).splitlines()[0]
        return ScreeningDecision(paper_id=paper_id, decision="uncertain", confidence=0.0,
                                 reasons=[f"sortie LLM hors contrat : {first}"], decider="llm")
    return ScreeningDecision(paper_id=paper_id, decider="llm", **judgment.model_dump())


def verify_evidence(decision: ScreeningDecision, paper: ScreeningPaper) -> ScreeningDecision:
    """TODO 5 - Ne croire que les preuves vérifiables dans le titre ou le résumé.

    1. Une citation introuvable dans le texte est retirée.
    2. Une seule citation inventée suffit : la décision devient "uncertain".
    3. Une inclusion sans aucune preuve devient "uncertain".
    """
    text = normalize(paper_text(paper))
    checked = decision.model_copy(deep=True)
    kept = [e for e in checked.evidence if normalize(e.quote) in text]
    invented = [e.quote for e in checked.evidence if normalize(e.quote) not in text]
    checked.evidence = kept
    if invented:
        checked.decision = "uncertain"
        checked.confidence = min(checked.confidence, 0.5)
        checked.reasons.append(f"citation introuvable : « {invented[0]} »")
    elif checked.decision == "include" and not kept:
        checked.decision = "uncertain"
        checked.confidence = min(checked.confidence, 0.5)
        checked.reasons.append("inclusion sans preuve")
    return checked


def llm_screen(model, paper: ScreeningPaper, protocol: Protocol) -> ScreeningDecision:
    """Code fourni : le LLM propose un jugement ; le logiciel le contrôle."""
    try:
        text = call_llm(model, build_screening_prompt(paper, protocol))
    except Exception as error:                     # réseau, clé, quota : le LLM se tait
        return ScreeningDecision(paper_id=paper.id, decision="uncertain", confidence=0.0,
                                 reasons=[f"LLM indisponible : {error.__class__.__name__}"], decider="llm")
    return verify_evidence(parse_judgment(text, paper.id), paper)


# ===========================================================================
# Partie C - Décider, s'abstenir, escalader
# ===========================================================================

def route(decision: ScreeningDecision, threshold: float = DEFAULT_THRESHOLD) -> Literal["auto", "humain"]:
    """TODO 6 - Agir ou s'abstenir : la décision est-elle assez sûre pour être automatique ?"""
    if decision.decision == "uncertain":
        return "humain"
    required = threshold if decision.decision == "include" else min(1.0, threshold + EXCLUDE_MARGIN)
    return "auto" if decision.confidence >= required else "humain"


def escalation_reason(rules: ScreeningDecision, llm: ScreeningDecision, threshold: float) -> str:
    """Code fourni : pourquoi l'humain est sollicité."""
    if rules.decision != llm.decision:
        return f"désaccord : {rules.decision} (règles) / {llm.decision} (LLM)"
    if rules.decision == "uncertain":
        return "abstention des deux évaluateurs"
    return f"confiance insuffisante (seuil {threshold})"


def combine(paper: ScreeningPaper, rules: ScreeningDecision, llm: ScreeningDecision,
            threshold: float = DEFAULT_THRESHOLD) -> ScreeningRecord:
    """TODO 7 - Double évaluation : agir seulement si les deux avis concordent et sont sûrs."""
    if rules.decision == llm.decision and route(rules, threshold) == route(llm, threshold) == "auto":
        final = ScreeningDecision(
            paper_id=paper.id, decision=rules.decision,
            confidence=min(rules.confidence, llm.confidence),
            reasons=["accord des règles et du LLM"], evidence=rules.evidence + llm.evidence,
            decider="accord")
        return ScreeningRecord(paper_id=paper.id, title=paper.title, rules=rules, llm=llm,
                               route="auto", final=final)
    return ScreeningRecord(paper_id=paper.id, title=paper.title, rules=rules, llm=llm,
                           route="humain", escalation=escalation_reason(rules, llm, threshold))


def screen_papers(papers: list[ScreeningPaper], protocol: Protocol, model,
                  threshold: float = DEFAULT_THRESHOLD) -> ScreeningState:
    """Code fourni : deux avis par article, puis la politique de décision."""
    state = ScreeningState(protocol=protocol, threshold=threshold)
    for paper in papers:
        state.records.append(combine(paper, rule_screen(paper, protocol),
                                     llm_screen(model, paper, protocol), threshold))
    return state


def evaluate(state: ScreeningState, gold: dict[str, str]) -> dict:
    """TODO 8 - Mesurer les erreurs des décisions AUTOMATIQUES par rapport aux étiquettes humaines."""
    auto = [r for r in state.records if r.route == "auto" and r.final is not None]
    false_exclusions = sum(1 for r in auto if r.final.decision == "exclude" and gold[r.paper_id] == "include")
    false_inclusions = sum(1 for r in auto if r.final.decision == "include" and gold[r.paper_id] == "exclude")
    total = len(state.records)
    human = total - len(auto)
    return {
        "articles": total,
        "auto": len(auto),
        "humain": human,
        "charge_humaine": round(human / total, 2) if total else 0.0,
        "faux_exclus": false_exclusions,
        "faux_inclus": false_inclusions,
        "erreurs_auto": false_exclusions + false_inclusions,
    }


# ===========================================================================
# Partie D - Boucler avec l'humain
# ===========================================================================

def apply_human_review(state: ScreeningState, reviews: dict[str, str]) -> int:
    """TODO 9 - Intégrer les décisions humaines de la file d'attente.

    reviews : {paper_id: "include" | "exclude"}. Retourne le nombre de décisions appliquées.
    Une revue ne remplace jamais une décision déjà prise ; "uncertain" n'est pas une décision.
    """
    queue = {r.paper_id: r for r in state.human_queue()}
    for paper_id, choice in reviews.items():
        if paper_id not in queue:
            raise ValueError(f"{paper_id} n'est pas dans la file humaine")
        if choice not in ("include", "exclude"):
            raise ValueError(f"décision humaine invalide : {choice}")
    for paper_id, choice in reviews.items():
        queue[paper_id].final = ScreeningDecision(
            paper_id=paper_id, decision=choice, confidence=1.0,
            reasons=[f"décision humaine ({queue[paper_id].escalation})"], decider="humain")
    return len(reviews)


def rescreen(state: ScreeningState, threshold: float) -> ScreeningState:
    """Code fourni : rejouer la politique avec un autre seuil, sans rappeler le LLM."""
    papers = {r.paper_id: ScreeningPaper(id=r.paper_id, title=r.title) for r in state.records}
    new = ScreeningState(protocol=state.protocol, threshold=threshold)
    for r in state.records:
        new.records.append(combine(papers[r.paper_id], r.rules, r.llm, threshold))
    return new


def sweep_thresholds(state: ScreeningState, gold: dict[str, str],
                     thresholds: tuple[float, ...] = THRESHOLDS) -> list[dict]:
    """Code fourni : une ligne d'évaluation par seuil."""
    return [{"seuil": t, **evaluate(rescreen(state, t), gold)} for t in thresholds]


def choose_threshold(rows: list[dict]) -> float | None:
    """TODO 10 - Le seuil qui minimise la charge humaine SANS aucune fausse exclusion.

    À charge égale, garder le seuil le plus élevé (le plus prudent).
    Aucun seuil sûr : None (tout reste à l'humain).
    """
    safe = [row for row in rows if row["faux_exclus"] == 0]
    if not safe:
        return None
    best = min(safe, key=lambda row: (row["humain"], -row["seuil"]))
    return best["seuil"]


def automation_only(state: ScreeningState, evaluator: Literal["regles", "llm"]) -> ScreeningState:
    """Code fourni : référence « sans humain » - un seul évaluateur décide de tout.

    Sans humain, un article incertain n'est pas retenu : il devient une exclusion.
    """
    new = ScreeningState(protocol=state.protocol, threshold=0.0)
    for r in state.records:
        opinion = r.rules if evaluator == "regles" else r.llm
        decision = "exclude" if opinion.decision == "uncertain" else opinion.decision
        final = opinion.model_copy(update={"decision": decision})
        new.records.append(ScreeningRecord(paper_id=r.paper_id, title=r.title, rules=r.rules,
                                           llm=r.llm, route="auto", final=final))
    return new


def compare_policies(state: ScreeningState, gold: dict[str, str]) -> list[dict]:
    """Code fourni : automatisation complète vs double évaluation avec humain."""
    return [
        {"politique": "règles seules, sans humain", **evaluate(automation_only(state, "regles"), gold)},
        {"politique": "LLM seul, sans humain", **evaluate(automation_only(state, "llm"), gold)},
        {"politique": f"double évaluation + humain (seuil {state.threshold})", **evaluate(state, gold)},
    ]
