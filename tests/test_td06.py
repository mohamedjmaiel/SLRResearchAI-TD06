"""Tests du TD06 : un groupe de tests indépendant par TODO (ne pas modifier).

    python -m pytest -v -k "fourni"         code fourni (vert dès le départ)
    python -m pytest -v -k "todo_6_"        un TODO
    python -m pytest -v tests/acquis_td05   non-régression du TD05 (et tests/acquis_td04)

Chaque test « todo_n » n'exerce que le TODO n : les fonctions des autres TODO
sont remplacées par des versions contrôlées (monkeypatch) lorsque c'est utile.
Seule exception : test_todo_10_integration_jeu_evaluation exécute toute la
chaîne et n'est vert qu'une fois les TODO 1 à 10 réalisés.
"""
import json

import pytest

import slrresearch.agentic_rag as agentic
from slrresearch.agentic_artifacts import (
    annotation_for,
    load_eval_set,
    load_memory,
    save_agentic_evidence,
    save_memory,
)
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
from slrresearch.agentic_rag import (
    ADVANCED,
    NAIVE,
    assess_passages,
    check_action,
    covered_concepts,
    decide_next,
    evaluate_run,
    recall,
    record_search,
    remember,
    rewrite_query,
    run_agent,
)
from slrresearch.agentic_tools import (
    MAX_K,
    MAX_RETRIEVALS,
    STRATEGIES,
    ActionRefused,
    Toolbox,
    concepts_from_question,
    concepts_from_td02,
    contains_term,
    make_decomposer,
    make_llm_decider,
    validate_action,
)
from slrresearch.demo_embeddings import DemoEmbeddings
from slrresearch.demo_llm import DemoChatModel
from slrresearch.rag_artifacts import load_rag_handoff
from slrresearch.rag_pipeline import build_corpus, build_llm_extractor, make_retrievers
from slrresearch.search_plan import ConceptGroup, ConceptExtraction

from .conftest import (
    COST,
    HYBRID_R,
    HYBRID_R6,
    LEXICAL_R,
    OTHER_PASSAGE,
    READ,
    READ_PASSAGE,
    SEMANTIC_Q,
    VOTE,
    VOTE_PASSAGE,
    evidence,
    make_state,
    passage,
)

W1001 = "https://openalex.org/W1001"
S2 = "s2-7f3a"
ALLOWED = {"lexical", "bm25", "semantique", "hybride"}


def search(strategy: Strategy, focus: int = 0) -> AgentAction:
    return AgentAction(action="search", focus=focus, retriever=strategy.retriever,
                       query_kind=strategy.query_kind, k=strategy.k)


def assessed(focus: Focus, status: str, missing=()) -> Focus:
    focus.assessment = Assessment(status=status, missing=list(missing))
    return focus


@pytest.fixture(scope="module")
def retrievers():
    documents, _ = load_rag_handoff()
    return make_retrievers(build_corpus(documents), DemoEmbeddings())


def demo_toolbox(retrievers, memory=None, behaviour="fidele", scenario="discipline") -> Toolbox:
    model = DemoChatModel(react_scenario=scenario, rag_behaviour=behaviour)
    return Toolbox(retrievers=retrievers, extractor=build_llm_extractor(model),
                   decomposer=make_decomposer(model), memory=memory)


# ===========================================================================
# Code fourni (vert dès le départ)
# ===========================================================================

def test_fourni_contains_term_mot_entier():
    assert contains_term("reached a high RECALL.", "recall")
    assert contains_term("zero-shot chain of thought", "chain of thought")
    assert not contains_term("every disagreement was settled", "agreement")
    assert not contains_term("recalled", "recall")


def test_fourni_concepts_du_vocabulaire_controle():
    labels = [c.label for c in concepts_from_question(
        "Quelle charge de lecture le screening par LLM économise-t-il et combien coûte le vote majoritaire ?")]
    assert labels == ["charge_de_lecture", "cout", "vote_majoritaire"]
    assert concepts_from_question("Quel temps fait-il ?") == []
    assert concepts_from_question("Quelle sensibilité ?")[0].terms == ["recall", "sensitivity"]


def test_fourni_concepts_td02_propres_a_la_rq():
    def rq(*labels):
        return ConceptExtraction(concepts=[ConceptGroup(label=l, terms=[l]) for l in labels])
    rq1, rq2 = rq("screening", "recall"), rq("screening", "human oversight")
    assert [c.label for c in concepts_from_td02(rq1, [rq2])] == ["recall"]
    assert [c.label for c in concepts_from_td02(rq1, [rq1])] == ["screening", "recall"]   # rien de propre


def test_fourni_contrat_agent_action():
    assert validate_action({"action": "search", "retriever": "bm25", "query_kind": "question", "k": 3}).k == 3
    with pytest.raises(ActionRefused) as error:
        validate_action({"action": "search", "retriever": "bm25"})
    assert error.value.status == "invalide"
    with pytest.raises(ActionRefused):
        validate_action({"action": "answer", "retriever": "bm25", "query_kind": "question", "k": 3})
    with pytest.raises(ActionRefused):
        validate_action({"action": "delete_corpus"})
    # Bien formée mais à refuser par le logiciel : le schéma ne connaît pas les limites.
    assert validate_action({"action": "search", "retriever": "google", "query_kind": "question", "k": 50})


def test_fourni_echelle_des_strategies_commence_comme_le_rag_naif():
    assert STRATEGIES[0] == NAIVE
    assert len(set(STRATEGIES)) == len(STRATEGIES)
    assert all(strategy.k <= MAX_K for strategy in STRATEGIES)
    assert ADVANCED in STRATEGIES


def test_fourni_decomposeur_de_demonstration():
    decompose = make_decomposer(DemoChatModel())
    assert decompose("Quelle charge de lecture le screening économise-t-il et combien coûte le vote majoritaire ?") \
        == ["Quelle charge de lecture le screening économise-t-il ?", "Combien coûte le vote majoritaire ?"]
    assert decompose("Qui prend la décision finale ?") == ["Qui prend la décision finale ?"]


def test_fourni_decideur_llm_indiscipline_propose_des_actions_a_refuser():
    decide = make_llm_decider(DemoChatModel(react_scenario="indiscipline"), sorted(ALLOWED))
    state = make_state()
    first = decide(state)
    assert first.action == "answer" and state.llm_calls == 1     # chaque décision coûte un appel LLM
    state.steps = 1
    assert decide(state).retriever == "google"
    state.steps = 2
    with pytest.raises(ActionRefused):
        decide(state)                                           # texte libre : hors contrat
    state.steps = 3
    assert decide(state).k == 50


def test_fourni_jeu_evaluation_et_annotations():
    items = load_eval_set()
    assert [item.id for item in items] == [f"E{n}" for n in range(1, 9)]
    assert {item.kind for item in items} >= {"facile", "vocabulaire", "multi_aspects", "hors_corpus", "paraphrase"}
    e3 = annotation_for("  quelle charge de lecture le screening par LLM économise-t-il et combien coûte le "
                        "vote majoritaire ?")
    assert e3.aspects == 2 and (W1001, 3) in e3.acceptable_pages
    assert annotation_for("Question absente ?") is None
    assert all(concepts_from_question(item.question) for item in items)


def test_fourni_rag_naif_repond_hors_corpus(retrievers):
    """Limite du RAG classique : une citation exacte… d'un passage qui ne répond pas."""
    e4 = next(item for item in load_eval_set() if item.id == "E4")
    state = agentic.run_fixed(e4.question, None, demo_toolbox(retrievers), NAIVE, "naive")
    assert state.outcome == "repondu"
    assert state.focuses[0].evidence.status == "verified"
    assert (state.focuses[0].evidence.paper_id, state.focuses[0].evidence.page) == (W1001, 4)


def test_fourni_rag_naif_s_abstient_sur_un_mot_hors_lexique(retrievers):
    e2 = next(item for item in load_eval_set() if item.id == "E2")
    state = agentic.run_fixed(e2.question, None, demo_toolbox(retrievers), NAIVE, "naive")
    assert state.outcome == "abstention" and state.llm_calls == 0 and state.retriever_calls == 1


def test_fourni_memoire_persistee(tmp_path):
    path = tmp_path / "agent_memory.json"
    assert load_memory(path).episodes == []
    memory = AgentMemory(episodes=[Episode(question="q", concepts=["cout"], strategy=LEXICAL_R)])
    save_memory(memory, path)
    assert load_memory(path) == memory


def test_fourni_artefact_preuves_par_focus(tmp_path):
    first = Focus(question="A ?", concepts=[READ], evidence=evidence(page=3, paper_id="P1", chunk_id="P1_p3_c2"),
                  winning=LEXICAL_R)
    second = Focus(question="B ?", concepts=[COST], evidence=evidence(status="human_review"))
    state = make_state(first, second)
    path = tmp_path / "verified_evidence.json"
    records = save_agentic_evidence(state, "sujet", path)
    assert [r["authorized_for_writing"] for r in records] == [True, False]
    assert records[0]["strategy"] == LEXICAL_R.label()
    assert json.loads(path.read_text(encoding="utf-8"))["origin"] == "RAG_TD06"


# ===========================================================================
# TODO 1 - covered_concepts
# ===========================================================================

def test_todo_1_concepts_couverts_dans_l_ordre():
    text = "Majority voting multiplied the cost by five."
    assert covered_concepts(text, [VOTE, READ, COST]) == ["vote_majoritaire", "cout"]


def test_todo_1_un_synonyme_suffit():
    assert covered_concepts("It reached 4.10 dollars.", [COST]) == ["cout"]


def test_todo_1_mot_entier_et_casse():
    assert covered_concepts("REVIEWERS HAD TO READ fewer abstracts", [READ]) == ["charge_de_lecture"]
    assert covered_concepts("costly decisions", [COST]) == []


# ===========================================================================
# TODO 2 - assess_passages
# ===========================================================================

@pytest.fixture
def coverage(monkeypatch):
    """covered_concepts (TODO 1) remplacé par une version de référence."""
    def reference(text, concepts):
        return [c.label for c in concepts if any(contains_term(text, term) for term in c.terms)]
    monkeypatch.setattr(agentic, "covered_concepts", reference)


def test_todo_2_suffisante(coverage):
    result = assess_passages([OTHER_PASSAGE, VOTE_PASSAGE], [COST, VOTE])
    assert result == Assessment(status="suffisante", best_chunk_id="P2_p3_c2",
                                covered=["cout", "vote_majoritaire"], missing=[])


def test_todo_2_partielle_et_premier_passage_a_egalite(coverage):
    cheap = passage("A", "The cost was low.")
    other = passage("B", "Dollars were spent.")
    result = assess_passages([cheap, other, READ_PASSAGE], [COST, VOTE])
    assert (result.status, result.best_chunk_id, result.covered, result.missing) == \
        ("partielle", "A", ["cout"], ["vote_majoritaire"])


def test_todo_2_insuffisante(coverage):
    assert assess_passages([], [COST]) == Assessment(status="insuffisante", missing=["cout"])
    result = assess_passages([OTHER_PASSAGE], [COST, VOTE])
    assert (result.status, result.best_chunk_id, result.missing) == ("insuffisante", None, ["cout", "vote_majoritaire"])


def test_todo_2_sans_concept_on_ne_juge_pas(coverage):
    with pytest.raises(ValueError):
        assess_passages([VOTE_PASSAGE], [])


# ===========================================================================
# TODO 3 - rewrite_query
# ===========================================================================

def test_todo_3_termes_dans_l_ordre():
    assert rewrite_query([COST, VOTE]) == "cost dollars majority voting"


def test_todo_3_sans_doublon_insensible_a_la_casse():
    duplicated = Concept(label="llm", terms=["LLM", "large language models"])
    again = Concept(label="modele", terms=["llm", "model"])
    assert rewrite_query([duplicated, again]) == "LLM large language models model"


def test_todo_3_liste_vide():
    assert rewrite_query([]) == ""


# ===========================================================================
# TODO 4 - record_search
# ===========================================================================

def fake_assess(passages, concepts):
    found = any("majority" in p["text"].lower() for p in passages)
    return Assessment(status="suffisante" if found else "insuffisante")


def test_todo_4_ajoute_les_nouveaux_passages_sans_retrier(monkeypatch):
    monkeypatch.setattr(agentic, "assess_passages", fake_assess)
    focus = Focus(question="q", concepts=[COST], passages=[dict(OTHER_PASSAGE)])
    low = {**VOTE_PASSAGE, "retrieval_score": 0.01}
    new = record_search(focus, LEXICAL_R, [dict(OTHER_PASSAGE), low])
    assert new == 1
    assert [p["chunk_id"] for p in focus.passages] == ["P1_p5_c1", "P2_p3_c2"]
    assert focus.tried == [LEXICAL_R]


def test_todo_4_reevalue_et_note_la_strategie_gagnante(monkeypatch):
    monkeypatch.setattr(agentic, "assess_passages", fake_assess)
    focus = Focus(question="q", concepts=[COST])
    record_search(focus, SEMANTIC_Q, [])
    assert focus.assessment.status == "insuffisante" and focus.winning is None
    record_search(focus, LEXICAL_R, [VOTE_PASSAGE])
    assert focus.assessment.status == "suffisante" and focus.winning == LEXICAL_R
    record_search(focus, HYBRID_R, [READ_PASSAGE])
    assert focus.winning == LEXICAL_R                      # la première stratégie gagnante reste
    assert focus.tried == [SEMANTIC_Q, LEXICAL_R, HYBRID_R]


def test_todo_4_retourne_zero_si_rien_de_nouveau(monkeypatch):
    monkeypatch.setattr(agentic, "assess_passages", fake_assess)
    focus = Focus(question="q", concepts=[COST], passages=[dict(VOTE_PASSAGE)])
    assert record_search(focus, HYBRID_R6, [dict(VOTE_PASSAGE)]) == 0
    assert len(focus.passages) == 1


# ===========================================================================
# TODO 5 - check_action
# ===========================================================================

def refused(action, state, allowed=ALLOWED) -> str:
    with pytest.raises(ActionRefused) as error:
        check_action(action, state, allowed)
    return str(error.value)


def test_todo_5_focus_inexistant(state):
    assert "focus" in refused(search(SEMANTIC_Q, focus=3), state).lower()
    assert refused(AgentAction(action="answer", focus=1), state)


def test_todo_5_retriever_et_k(state):
    check_action(search(SEMANTIC_Q), state, ALLOWED)                     # autorisée : ne lève rien
    refused(AgentAction(action="search", retriever="google", query_kind="question", k=3), state)
    refused(search(SEMANTIC_Q), state, allowed={"bm25"})
    refused(AgentAction(action="search", retriever="bm25", query_kind="question", k=MAX_K + 1), state)
    check_action(AgentAction(action="search", retriever="bm25", query_kind="question", k=MAX_K), state, ALLOWED)


def test_todo_5_repetition_par_focus_et_budget():
    first = Focus(question="A ?", concepts=[COST], tried=[SEMANTIC_Q])
    second = Focus(question="B ?", concepts=[VOTE])
    state = make_state(first, second)
    refused(search(SEMANTIC_Q, focus=0), state)
    check_action(search(SEMANTIC_Q, focus=1), state, ALLOWED)            # même stratégie, autre focus
    state.retriever_calls = MAX_RETRIEVALS
    refused(search(LEXICAL_R, focus=1), state)


def test_todo_5_decompose_answer_abstain(state):
    check_action(AgentAction(action="decompose"), state, ALLOWED)
    state.decomposed = True
    refused(AgentAction(action="decompose"), state)
    refused(AgentAction(action="answer"), state)
    assessed(state.focuses[0], "suffisante")
    check_action(AgentAction(action="answer"), state, ALLOWED)
    check_action(AgentAction(action="abstain"), make_state(), ALLOWED)


# ===========================================================================
# TODO 6 - decide_next
# ===========================================================================

def strategy_of(action: AgentAction) -> Strategy:
    assert action.action == "search"
    return action.strategy()


def test_todo_6_commence_comme_le_rag_naif(state):
    action = decide_next(state)
    assert (action.focus, strategy_of(action)) == (0, STRATEGIES[0])
    assert action.reason


def test_todo_6_echelle_puis_suggestion_memoire_d_abord():
    focus = assessed(Focus(question="q", concepts=[COST], tried=[SEMANTIC_Q]), "insuffisante")
    assert strategy_of(decide_next(make_state(focus))) == LEXICAL_R
    hinted = Focus(question="q", concepts=[COST], hint=HYBRID_R)
    assert strategy_of(decide_next(make_state(hinted))) == HYBRID_R
    hinted.tried = [HYBRID_R]
    assert strategy_of(decide_next(make_state(hinted))) == SEMANTIC_Q    # suggestion déjà essayée


def test_todo_6_decompose_une_seule_fois_si_partielle():
    focus = assessed(Focus(question="q", concepts=[COST, VOTE], tried=[SEMANTIC_Q]), "partielle", ["vote_majoritaire"])
    action = decide_next(make_state(focus))
    assert (action.action, action.focus) == ("decompose", 0)
    assert strategy_of(decide_next(make_state(focus, decomposed=True))) == LEXICAL_R


def test_todo_6_passe_au_focus_ouvert_suivant():
    done = assessed(Focus(question="A ?", concepts=[READ], tried=[SEMANTIC_Q]), "suffisante")
    todo = assessed(Focus(question="B ?", concepts=[COST], tried=[SEMANTIC_Q]), "insuffisante")
    action = decide_next(make_state(done, todo, decomposed=True))
    assert (action.focus, strategy_of(action)) == (1, LEXICAL_R)


def test_todo_6_repondre_ou_s_abstenir_quand_tout_est_essaye():
    exhausted = assessed(Focus(question="B ?", concepts=[COST], tried=list(STRATEGIES)), "insuffisante")
    assert decide_next(make_state(exhausted)).action == "abstain"
    done = assessed(Focus(question="A ?", concepts=[READ], tried=[SEMANTIC_Q]), "suffisante")
    assert decide_next(make_state(done, exhausted, decomposed=True)).action == "answer"
    assert decide_next(make_state(done)).action == "answer"


# ===========================================================================
# TODO 7 - run_agent
# ===========================================================================

class Script:
    """Décideur scripté : rejoue des propositions (dict ou AgentAction)."""

    def __init__(self, *proposals):
        self.proposals, self.calls = list(proposals), 0

    def __call__(self, state):
        proposal = self.proposals[min(self.calls, len(self.proposals) - 1)]
        self.calls += 1
        return proposal


@pytest.fixture
def isolated(monkeypatch):
    """check_action et execute contrôlés : seul run_agent est testé."""
    def fake_check(action, state, retrievers):
        if action.action == "search" and action.retriever not in retrievers:
            raise ActionRefused(f"Retriever non autorisé : {action.retriever}")

    def fake_execute(action, state, toolbox):
        return f"exécuté : {action.action}"

    monkeypatch.setattr(agentic, "check_action", fake_check)
    monkeypatch.setattr(agentic, "execute", fake_execute)
    return Toolbox(retrievers={"bm25": None}, extractor=None)


def test_todo_7_decider_controler_agir_puis_s_arreter(isolated):
    decide = Script(search(Strategy(retriever="bm25", query_kind="question", k=3)), AgentAction(action="abstain"),
                    AgentAction(action="answer"))
    state = run_agent(make_state(), decide, isolated)
    assert decide.calls == 2
    assert [(t.step, t.status, t.action["action"]) for t in state.trace] == [(1, "ok", "search"), (2, "ok", "abstain")]
    assert state.trace[0].observation == "exécuté : search"
    assert state.outcome == "abstention"                         # finalize a été appelé


def test_todo_7_refus_et_sortie_invalide_consomment_une_etape(isolated):
    decide = Script({"action": "search", "retriever": "google", "query_kind": "question", "k": 3},
                    {"action": "search"}, AgentAction(action="abstain"))
    state = run_agent(make_state(), decide, isolated)
    assert [t.status for t in state.trace] == ["refusee", "invalide", "ok"]
    assert "google" in state.trace[0].observation
    assert state.trace[1].action is None and state.steps == 3


def test_todo_7_boucle_bornee(isolated):
    decide = Script({"action": "search", "retriever": "google", "query_kind": "question", "k": 3})
    state = run_agent(make_state(), decide, isolated, max_steps=4)
    assert state.steps == 4 and decide.calls == 4
    assert state.outcome == "budget_epuise"


# ===========================================================================
# TODO 8 - remember
# ===========================================================================

def solved(status="verified", winning=LEXICAL_R, concepts=(VOTE, COST)) -> Focus:
    return Focus(question="Combien coûte le vote majoritaire ?", concepts=list(concepts),
                 evidence=evidence(status=status), winning=winning)


def test_todo_8_seules_les_preuves_verifiees_sont_memorisees():
    memory = AgentMemory()
    state = make_state(solved(), solved(status="human_review"), solved(status="rejected"), solved(winning=None))
    assert remember(memory, state) == 1
    assert memory.episodes == [Episode(question="Combien coûte le vote majoritaire ?",
                                       concepts=["cout", "vote_majoritaire"], strategy=LEXICAL_R)]


def test_todo_8_souvenir_identique_incremente_uses():
    memory = AgentMemory()
    remember(memory, make_state(solved()))
    assert remember(memory, make_state(solved())) == 0
    assert memory.episodes[0].uses == 2
    assert remember(memory, make_state(solved(winning=HYBRID_R))) == 1     # autre stratégie : autre souvenir
    assert len(memory.episodes) == 2


def test_todo_8_abstention_rien_n_est_ecrit():
    memory = AgentMemory()
    assert remember(memory, make_state(Focus(question="q", concepts=[COST]), outcome="abstention")) == 0
    assert memory.episodes == []


# ===========================================================================
# TODO 9 - recall
# ===========================================================================

def episode(labels, strategy, uses=1):
    return Episode(question="q", concepts=sorted(labels), strategy=strategy, uses=uses)


def test_todo_9_memoire_vide_ou_absente():
    assert recall(None, [COST]) is None
    assert recall(AgentMemory(), [COST]) is None


def test_todo_9_similarite_de_jaccard_et_seuil():
    memory = AgentMemory(episodes=[episode(["cout", "vote_majoritaire"], LEXICAL_R)])
    assert recall(memory, [COST, VOTE]) == LEXICAL_R                           # 2/2 = 1
    assert recall(memory, [READ, COST, VOTE]) == LEXICAL_R                     # 2/3
    assert recall(memory, [COST]) == LEXICAL_R                                 # 1/2 = seuil
    assert recall(memory, [COST, READ]) is None                                # 1/3
    assert recall(memory, [COST, VOTE], threshold=1.01) is None


def test_todo_9_departage_uses_puis_le_plus_ancien():
    memory = AgentMemory(episodes=[episode(["cout"], SEMANTIC_Q), episode(["cout"], LEXICAL_R, uses=3),
                                   episode(["cout"], HYBRID_R, uses=3)])
    assert recall(memory, [COST]) == LEXICAL_R
    better = AgentMemory(episodes=[episode(["cout", "charge_de_lecture"], HYBRID_R, uses=9),
                                   episode(["cout"], SEMANTIC_Q)])
    assert recall(better, [COST]) == SEMANTIC_Q                                # similarité d'abord


# ===========================================================================
# TODO 10 - evaluate_run
# ===========================================================================

def annotation(expected="repondre", pages=(("P1", 3), ("P2", 3)), aspects=1) -> EvalQuestion:
    return EvalQuestion(id="T", question="q", kind="test", expected=expected,
                        acceptable_pages=list(pages), aspects=aspects)


def test_todo_10_aspects_exigent_des_pages_distinctes():
    same = make_state(solved(), solved())
    both = make_state(Focus(question="A", concepts=[READ], evidence=evidence(paper_id="P1", chunk_id="P1_p3_c2")),
                      solved())
    assert evaluate_run(same, annotation(aspects=2))["correct"] is False
    assert evaluate_run(both, annotation(aspects=2))["correct"] is True
    assert evaluate_run(same, annotation(aspects=1))["correct"] is True


def test_todo_10_abstention_attendue():
    state = make_state(Focus(question="q", concepts=[COST]), outcome="abstention")
    result = evaluate_run(state, annotation(expected="abstention", pages=()))
    assert result["correct"] is True and result["fidelite"] is None and result["pertinence"] is None
    review = make_state(solved(status="human_review"))
    assert evaluate_run(review, annotation(expected="abstention", pages=()))["correct"] is True
    assert evaluate_run(make_state(solved()), annotation(expected="abstention", pages=()))["correct"] is False


def test_todo_10_fidelite_pertinence_et_compteurs():
    wrong_page = Focus(question="A", concepts=[READ], evidence=evidence(page=5, chunk_id="P2_p5_c1"))
    state = make_state(solved(), wrong_page, solved(status="rejected"), llm_calls=4, retriever_calls=3,
                       steps=6, outcome="partiel")
    result = evaluate_run(state, annotation())
    assert result == {"correct": True, "fidelite": 2 / 3, "pertinence": 1 / 2, "appels_llm": 4,
                      "appels_retriever": 3, "pas": 6, "issue": "partiel"}


def test_todo_10_integration_jeu_evaluation(retrievers):
    """Toute la chaîne (TODO 1 à 10) : naïf 1/8, avancé 5/8, agentique 8/8 ; la mémoire réduit le coût."""
    rows, memory = [], AgentMemory()
    for item in load_eval_set():
        for config in ("naive", "advanced", "agentique", "agentique+memoire"):
            if config in ("naive", "advanced"):
                state = agentic.run_fixed(item.question, None, demo_toolbox(retrievers),
                                          NAIVE if config == "naive" else ADVANCED, config)
            else:
                state = agentic.run_agentic(item.question, None,
                                            demo_toolbox(retrievers, memory if config.endswith("memoire") else None),
                                            config=config)
            rows.append({"config": config, **evaluate_run(state, item)})
    table = {row["config"]: row for row in agentic.summarize(rows)}
    assert [table[c]["correctes"] for c in ("naive", "advanced", "agentique", "agentique+memoire")] == [1, 5, 8, 8]
    assert table["agentique"]["appels_retriever"] > table["agentique+memoire"]["appels_retriever"]
    assert table["agentique"]["appels_llm"] > table["advanced"]["appels_llm"]
    assert {episode.strategy for episode in memory.episodes} >= {LEXICAL_R}
