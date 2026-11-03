"""TD06 - RAG agentique : décider, chercher autrement, s'abstenir, mémoriser, mesurer.

    RAG naïf      question ─→ retriever fixe ─→ Top-k ─→ LLM ─→ contrôle            (TD05)
    RAG avancé    question ─→ reformulation ─→ retriever hybride ─→ Top-k ─→ LLM ─→ contrôle
    RAG agentique boucle bornée : décider ─→ contrôler l'action ─→ agir ─→ OBSERVER ─→ ...
                  actions : chercher (stratégie), décomposer, répondre, s'abstenir

Observer   TODO 1 et 2     concepts couverts, preuves suffisantes ?          (guidé)
Agir       TODO 3 et 4     requête reformulée, enregistrer une recherche
Décider    TODO 5 à 7      garde-fous, politique codée, boucle bornée
Mémoriser  TODO 8 et 9     écrire et relire la mémoire épisodique
Évaluer    TODO 10         une exécution face à son annotation             (autonome)

Ce fichier est le seul à modifier. Le code marqué « fourni » est à lire.
"""
import time
from typing import Callable

from slrresearch.agentic_models import (
    AgentAction,
    AgentMemory,
    AgentState,
    Assessment,
    Concept,
    Episode,
    EvalQuestion,
    Focus,
    Strategy,
)
from slrresearch.agentic_tools import (
    MAX_K,
    MAX_RETRIEVALS,
    MAX_STEPS,
    MEMORY_THRESHOLD,
    STRATEGIES,
    ActionRefused,
    Toolbox,
    concepts_from_question,
    contains_term,
    validate_action,
)
from slrresearch.rag_pipeline import MIN_SCORE, build_context, run_rag, verify_evidence

Decider = Callable[[AgentState], AgentAction]

NAIVE = Strategy(retriever="semantique", query_kind="question", k=3)      # TD05 : question brute
ADVANCED = Strategy(retriever="hybride", query_kind="reformulee", k=3)    # reformulation + hybride


# ===========================================================================
# Observer : les preuves retrouvées suffisent-elles ?
# ===========================================================================

def covered_concepts(text: str, concepts: list[Concept]) -> list[str]:
    """Libellés des concepts dont AU MOINS UN terme apparaît dans le texte (contains_term),
    dans l'ordre de la liste des concepts."""
    # TODO 1 - Parcourir les concepts ; garder le libellé si l'un de ses termes est présent.
    raise NotImplementedError("TODO 1 - covered_concepts")


def assess_passages(passages: list[dict], concepts: list[Concept]) -> Assessment:
    """Observation après une recherche : le MEILLEUR passage couvre-t-il tous les concepts ?

    - aucun concept requis : ValueError (on ne juge pas sans critère) ;
    - meilleur passage = celui qui couvre le plus de concepts ; à égalité, le premier de la liste ;
    - tous couverts      -> "suffisante"   (best_chunk_id, covered = tous, missing = [])
    - au moins un couvert -> "partielle"   (best_chunk_id, covered, missing)
    - aucun (ou aucun passage) -> "insuffisante" (best_chunk_id = None, missing = tous)
    """
    # TODO 2 - Refuser une liste de concepts vide ; trouver le meilleur passage ; produire l'Assessment.
    raise NotImplementedError("TODO 2 - assess_passages")


# ===========================================================================
# Agir : reformuler la requête, enregistrer le résultat d'une recherche
# ===========================================================================

def rewrite_query(concepts: list[Concept]) -> str:
    """Reformulation (pré-retrieval) : les termes du corpus de tous les concepts, dans l'ordre,
    sans doublon (comparaison insensible à la casse), séparés par une espace."""
    # TODO 3 - Concaténer les termes des concepts sans doublon.
    raise NotImplementedError("TODO 3 - rewrite_query")


def record_search(focus: Focus, strategy: Strategy, results: list[dict]) -> int:
    """Met à jour le focus après une recherche et retourne le nombre de passages NOUVEAUX.

    1. la stratégie rejoint focus.tried ;
    2. les passages dont le chunk_id n'est pas encore connu sont AJOUTÉS à la fin de focus.passages
       (les scores de retrievers différents ne sont pas comparables : on ne retrie pas) ;
    3. focus.assessment = assess_passages(focus.passages, focus.concepts) ;
    4. si les preuves deviennent suffisantes et que focus.winning est vide : winning = strategy.
    """
    # TODO 4 - Mémoriser la stratégie, ajouter les nouveaux passages, réévaluer, noter la stratégie gagnante.
    raise NotImplementedError("TODO 4 - record_search")


# ===========================================================================
# Décider : garde-fous, politique codée, boucle bornée
# ===========================================================================

def check_action(action: AgentAction, state: AgentState, retrievers: set[str]) -> None:
    """Le logiciel dispose : lève ActionRefused(motif) si l'action est interdite.

    Règles, dans l'ordre :
    1. action.focus doit désigner un focus existant ;
    2. search    : retriever autorisé ; k <= MAX_K ; budget MAX_RETRIEVALS non atteint ;
                   stratégie jamais essayée pour CE focus ;
    3. decompose : une seule fois par question (state.decomposed) ;
    4. answer    : au moins un focus aux preuves suffisantes ;
    5. abstain   : toujours autorisée.
    """
    # TODO 5 - Appliquer les règles dans l'ordre ; lever ActionRefused("motif") en cas de refus.
    raise NotImplementedError("TODO 5 - check_action")


def decide_next(state: AgentState) -> AgentAction:
    """Politique codée (niveau 2) : règles évaluées dans l'ordre.

    Stratégies restantes d'un focus : sa suggestion mémoire (focus.hint) en premier si elle
    n'a pas été essayée, puis STRATEGIES dans l'ordre, sans celles déjà essayées.
    Focus ouvert : preuves non suffisantes ET au moins une stratégie restante.

    R1  aucun focus ouvert : "answer" si un focus est suffisant, sinon "abstain" ;
    R2  premier focus ouvert ; si la question n'a pas encore été décomposée et que ses preuves
        sont "partielle" : "decompose" ce focus ;
    R3  sinon : "search" ce focus avec sa première stratégie restante.
    Chaque action porte une raison (reason) lisible dans la trace.
    """
    # TODO 6 - Calculer les stratégies restantes, trouver le premier focus ouvert, appliquer R1 à R3.
    raise NotImplementedError("TODO 6 - decide_next")


def run_agent(state: AgentState, decide: Decider, toolbox: Toolbox,
              max_steps: int = MAX_STEPS) -> AgentState:
    """Boucle bornée : décider -> valider (validate_action) -> contrôler (check_action) -> agir (execute).

    - au plus max_steps étapes ; une action refusée ou invalide est enregistrée
      (state.record(action, refusal.status, str(refusal))) et consomme une étape ;
    - une action acceptée est exécutée puis enregistrée avec le statut "ok" et l'observation retournée ;
    - la boucle s'arrête après "answer" ou "abstain" ;
    - à la fin : finalize(state), puis retourner l'état.
    """
    # TODO 7 - Écrire la boucle bornée ; gérer ActionRefused ; s'arrêter sur answer / abstain.
    raise NotImplementedError("TODO 7 - run_agent")


# ===========================================================================
# Mémoriser : mémoire épisodique (long terme)
# ===========================================================================

def remember(memory: AgentMemory, state: AgentState) -> int:
    """Écrit un souvenir par focus dont la preuve est VERIFIED et la stratégie gagnante connue.

    - concepts = libellés du focus, triés ;
    - souvenir identique déjà présent (mêmes concepts, même stratégie) : uses += 1 ;
    - sinon : nouvel Episode(question=focus.question, concepts=..., strategy=focus.winning) ;
    - human_review, rejected ou abstention : rien n'est écrit (pas de mémoire empoisonnée).
    Retourne le nombre de NOUVEAUX souvenirs.
    """
    # TODO 8 - Ne mémoriser que les preuves vérifiées ; incrémenter uses si le souvenir existe.
    raise NotImplementedError("TODO 8 - remember")


def recall(memory: AgentMemory | None, concepts: list[Concept],
           threshold: float = MEMORY_THRESHOLD) -> Strategy | None:
    """Stratégie du souvenir le plus proche, ou None.

    similarité = Jaccard(libellés du souvenir, libellés demandés) = |A ∩ B| / |A ∪ B| ;
    meilleur souvenir = similarité maximale, puis uses maximal, puis le plus ancien ;
    retourner sa stratégie si similarité >= threshold ; sinon (ou mémoire vide / absente) None.
    """
    # TODO 9 - Calculer la similarité de Jaccard, choisir le meilleur souvenir, appliquer le seuil.
    raise NotImplementedError("TODO 9 - recall")


# ===========================================================================
# Évaluer : une exécution face à son annotation
# ===========================================================================

def evaluate_run(state: AgentState, annotation: EvalQuestion) -> dict:
    """Mesures d'une exécution (naïve, avancée ou agentique) d'une question annotée.

    preuves     = preuves de tous les focus (focus.evidence non vide)
    verifiees   = preuves au statut "verified"
    pertinentes = pages DISTINCTES (paper_id, page) des preuves vérifiées qui sont acceptables
    - correct     : expected "abstention" -> aucune preuve vérifiée ;
                    expected "repondre"   -> au moins annotation.aspects pages pertinentes ;
    - fidelite    : verifiees / preuves (None si aucune preuve) ;
    - pertinence  : preuves vérifiées sur une page acceptable / verifiees (None si aucune) ;
    - appels_llm, appels_retriever, pas : compteurs de l'état ; issue : state.outcome.
    """
    # TODO 10 - Calculer correct, fidelite, pertinence et recopier les compteurs.
    raise NotImplementedError("TODO 10 - evaluate_run")


# ===========================================================================
# Code fourni : démarrer, exécuter une action, conclure
# ===========================================================================

def make_focus(question: str, concepts: list[Concept], memory: AgentMemory | None = None) -> Focus:
    """Un focus et, si une mémoire est fournie, sa suggestion (recall)."""
    return Focus(question=question, concepts=concepts, hint=recall(memory, concepts) if memory else None)


def start_state(question: str, concepts: list[Concept] | None = None, memory: AgentMemory | None = None,
                config: str = "agentique") -> AgentState:
    """État initial. Sans concepts fournis (TD02), on les lit dans le vocabulaire contrôlé."""
    concepts = concepts or concepts_from_question(question)
    if not concepts:
        raise ValueError("Aucun concept du vocabulaire contrôlé dans la question : "
                         "impossible de juger les preuves (complétez GLOSSARY ou passez par le TD02)")
    return AgentState(question=question, config=config, focuses=[make_focus(question, concepts, memory)])


def query_for(focus: Focus, query_kind: str) -> str:
    return focus.question if query_kind == "question" else rewrite_query(focus.concepts)


def select_context(focus: Focus, k: int) -> list[dict]:
    """Post-retrieval : seuls les passages qui couvrent au moins un concept sont transmis au LLM,
    ceux qui en couvrent le plus d'abord, puis dans l'ordre de découverte (k au plus)."""
    scored = [(len(covered_concepts(passage["text"], focus.concepts)), position, passage)
              for position, passage in enumerate(focus.passages)]
    ranked = sorted((item for item in scored if item[0] > 0), key=lambda item: (-item[0], item[1]))
    return [passage for _, _, passage in ranked[:k]]


def answer_focus(focus: Focus, toolbox: Toolbox) -> None:
    """Génération + contrôle du TD05 sur les passages du focus.

    La question est transmise avec les termes du corpus (requête reformulée) : le corpus est
    en anglais, la question en français ; le générateur sait ainsi quelle phrase citer."""
    selected = select_context(focus, toolbox.answer_k)
    question = f"{focus.question} (termes du corpus : {rewrite_query(focus.concepts)})"
    proposal = toolbox.extractor(question, build_context(selected))
    focus.evidence = verify_evidence(proposal, selected)
    focus.context_chunk_ids = [passage["chunk_id"] for passage in selected]


def execute(action: AgentAction, state: AgentState, toolbox: Toolbox) -> str:
    """Exécute une action AUTORISÉE et retourne l'observation (texte de la trace)."""
    focus = state.focuses[action.focus]
    if action.action == "search":
        strategy = action.strategy()
        query = query_for(focus, strategy.query_kind)
        results = toolbox.retrievers[strategy.retriever](query, strategy.k)
        state.retriever_calls += 1
        new = record_search(focus, strategy, results)
        missing = f" ; manque : {', '.join(focus.assessment.missing)}" if focus.assessment.missing else ""
        return (f"{strategy.label()} « {query[:70]} » : {len(results)} passage(s), {new} nouveau(x) ; "
                f"preuves {focus.assessment.status}{missing}")
    if action.action == "decompose":
        state.decomposed = True
        if toolbox.decomposer is None:
            return "Décomposition indisponible : on poursuit sur la question entière"
        state.llm_calls += 1
        subs = []
        for question in toolbox.decomposer(focus.question):
            concepts = concepts_from_question(question) or []
            if concepts:
                sub = make_focus(question, concepts, toolbox.memory)
                sub.passages = [dict(passage) for passage in focus.passages]   # déjà retrouvés
                sub.assessment = assess_passages(sub.passages, sub.concepts)
                subs.append(sub)
        if len(subs) < 2:
            return "Question non décomposable : on poursuit sur la question entière"
        state.focuses = subs
        return "Décomposée en " + " | ".join(f"[{i}] {s.question} ({s.assessment.status})"
                                            for i, s in enumerate(subs))
    if action.action == "answer":
        answered = []
        for index, item in enumerate(state.focuses):
            if item.sufficient and item.evidence is None:
                answer_focus(item, toolbox)
                state.llm_calls += 1
                answered.append(f"[{index}] {item.evidence.status} ({item.evidence.chunk_id}, p. {item.evidence.page})")
        gaps = [str(i) for i, item in enumerate(state.focuses) if not item.sufficient]
        gap = f" ; focus sans preuve : {', '.join(gaps)}" if gaps else ""
        return "Réponse : " + ", ".join(answered) + gap
    return "Abstention : aucune preuve suffisante, le LLM n'est pas appelé pour répondre"


def finalize(state: AgentState) -> AgentState:
    """Issue de l'exécution, calculée par le programme."""
    evidences = [focus.evidence for focus in state.focuses if focus.evidence is not None]
    verified = [evidence for evidence in evidences if evidence.status == "verified"]
    last = next((step for step in reversed(state.trace) if step.status == "ok"), None)
    if evidences:
        state.outcome = ("repondu" if len(verified) == len(state.focuses) else
                         "partiel" if verified else "sans_preuve")
    elif last and last.action and last.action["action"] == "abstain":
        state.outcome = "abstention"
    else:
        state.outcome = "budget_epuise"
    return state


# ===========================================================================
# Code fourni : les configurations comparées (même question, même corpus, même contrôle)
# ===========================================================================

def run_fixed(question: str, concepts: list[Concept] | None, toolbox: Toolbox, strategy: Strategy,
              config: str) -> AgentState:
    """RAG naïf ou avancé : une stratégie FIXE, un retrieval, une génération (run_rag du TD05)."""
    started = time.perf_counter()
    state = start_state(question, concepts, config=config)
    focus = state.focuses[0]
    query = query_for(focus, strategy.query_kind)
    rag = run_rag(question, query, toolbox.retrievers[strategy.retriever], toolbox.extractor,
                  k=strategy.k, min_score=MIN_SCORE[strategy.retriever])
    state.retriever_calls = 1
    focus.tried, focus.passages = [strategy], rag.selected_chunks
    action = AgentAction(action="search", retriever=strategy.retriever, query_kind=strategy.query_kind,
                         k=strategy.k, reason="pipeline fixe")
    state.record(action, "ok", f"{strategy.label()} « {query[:70]} » : {len(rag.selected_chunks)} passage(s)")
    if rag.abstained:
        state.record(AgentAction(action="abstain", reason="seuil de score"), "ok", rag.reason)
    else:
        state.llm_calls = 1
        focus.evidence, focus.context_chunk_ids = rag.evidence, [c["chunk_id"] for c in rag.selected_chunks]
        state.record(AgentAction(action="answer", reason="pipeline fixe"), "ok",
                     f"Réponse : {rag.evidence.status} ({rag.evidence.chunk_id}, p. {rag.evidence.page})")
    finalize(state)
    state.duration_ms = (time.perf_counter() - started) * 1000
    return state


def run_agentic(question: str, concepts: list[Concept] | None, toolbox: Toolbox, decide: Decider = None,
                config: str = "agentique", learn: bool = True) -> AgentState:
    """RAG agentique : état initial (avec la mémoire de la boîte à outils), boucle, puis mémorisation."""
    started = time.perf_counter()
    state = start_state(question, concepts, toolbox.memory, config=config)
    run_agent(state, decide or decide_next, toolbox)
    if learn and toolbox.memory is not None:
        remember(toolbox.memory, state)
    state.duration_ms = (time.perf_counter() - started) * 1000
    return state


def summarize(rows: list[dict]) -> list[dict]:
    """Moyennes par configuration (lignes produites par evaluate_run + clé "config")."""
    table = []
    for config in dict.fromkeys(row["config"] for row in rows):
        group = [row for row in rows if row["config"] == config]

        def mean(key):
            values = [row[key] for row in group if row[key] is not None]
            return round(sum(values) / len(values), 2) if values else None

        table.append({"config": config, "questions": len(group),
                      "correctes": sum(row["correct"] for row in group),
                      "fidelite": mean("fidelite"), "pertinence": mean("pertinence"),
                      "appels_llm": mean("appels_llm"), "appels_retriever": mean("appels_retriever"),
                      "pas": mean("pas")})
    return table
