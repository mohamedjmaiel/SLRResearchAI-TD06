"""TD06 - Interface : RAG naïf, avancé et agentique sur le même corpus.

    python -m streamlit run streamlit_app.py

1. Sujet → TD02 : RQ, concepts proposés, revue humaine
2. Corpus du TD05 (TD03 ∩ TD04 ∩ documents), chunks, index (acquis)
3. Une question : naïf / avancé / agentique, trace de l'agent       TODO 1-7
4. Mémoire épisodique                                               TODO 8-9
5. Évaluation expérimentale                                         TODO 10
"""
import streamlit as st

from slrresearch.agentic_artifacts import (
    annotation_for,
    load_eval_set,
    load_memory,
    save_agentic_evidence,
    save_memory,
    save_report,
)
from slrresearch.agentic_models import AgentMemory
from slrresearch.agentic_rag import ADVANCED, NAIVE, evaluate_run, run_agentic, run_fixed, summarize
from slrresearch.agentic_tools import (
    Toolbox,
    concepts_from_question,
    concepts_from_td02,
    make_decomposer,
    make_llm_decider,
)
from slrresearch.config import Settings, build_embeddings, build_model, resolve_decoding
from slrresearch.demo_embeddings import DemoEmbeddings
from slrresearch.demo_llm import RAG_BEHAVIOURS, REACT_SCENARIOS, DemoChatModel
from slrresearch.models import short_error
from slrresearch.rag_artifacts import load_rag_handoff
from slrresearch.rag_pipeline import build_corpus, build_llm_extractor, make_retrievers
from slrresearch.search_plan import apply_human_review, format_concepts_text, parse_concepts_text
from slrresearch.td02_bridge import (
    CONCEPT_MAX_OUTPUT_TOKENS,
    REFERENCE_TOPIC,
    generate_questions_for_topic,
    propose_concepts_for_topic,
)

CONFIG_LABELS = {"naive": "Naïf (TD05)", "advanced": "Avancé (reformulation + hybride)",
                 "agentique": "Agentique (politique codée)", "agentique+memoire": "Agentique + mémoire",
                 "agentique-llm": "Agentique (décideur LLM)"}
STATUS_ICONS = {"ok": "✅", "refusee": "⛔", "invalide": "⚠️"}

st.set_page_config(page_title="TD06 · RAG agentique", page_icon="🧭", layout="wide")
st.title("TD06 — RAG agentique : décider, chercher autrement, s'abstenir, mesurer")
st.caption("même corpus que le TD05 · décider → contrôler l'action → agir → observer · "
           "mémoire épisodique · comparaison naïf / avancé / agentique")


def attempt(step: str, action):
    """Exécute une étape ; un TODO manquant arrête l'affichage avec un message clair."""
    try:
        return action()
    except NotImplementedError as error:
        st.warning(f"**{step}** non disponible : {error}. Complétez ce TODO, puis relancez l'action.")
    except Exception as error:  # clé absente, Ollama arrêté, contrat violé...
        st.error(f"**{step}** impossible : {short_error(error)}")
    st.stop()


with st.sidebar:
    st.header("Paramètres")
    llm_mode = st.radio("LLM", ["demo", "reel"], horizontal=True,
                        format_func=lambda v: "Démonstration" if v == "demo" else "Réel (.env)")
    behaviour = st.selectbox("Extraction (LLM de démonstration)", RAG_BEHAVIOURS, disabled=llm_mode != "demo")
    embedding_mode = st.radio("Embeddings", ["demo", "ollama"], horizontal=True,
                              format_func=lambda v: "Démonstration" if v == "demo" else "Ollama")
    st.divider()
    decider_mode = st.radio("Décideur de l'agent", ["regles", "llm"],
                            format_func=lambda v: "Politique codée (niveau 2)" if v == "regles"
                            else "LLM (niveau 3)")
    scenario = st.selectbox("Scénario du décideur LLM de démonstration", REACT_SCENARIOS,
                            disabled=decider_mode != "llm" or llm_mode != "demo")


def models():
    if llm_mode == "demo":
        model = DemoChatModel(react_scenario=scenario, rag_behaviour=behaviour)
        return model, model, model, model
    settings = Settings()
    return (build_model(settings, resolve_decoding("greedy")),
            build_model(settings, resolve_decoding("greedy", max_output_tokens=CONCEPT_MAX_OUTPUT_TOKENS)),
            build_model(settings, resolve_decoding("greedy", max_output_tokens=800)),
            build_model(settings, resolve_decoding("greedy", max_output_tokens=300)))


# ---------------------------------------------------------------------------
# 1. Sujet → TD02
# ---------------------------------------------------------------------------
st.header("1. Sujet → TD02 : questions et concepts")
with st.form("topic"):
    topic = st.text_input("Sujet de la revue", value=REFERENCE_TOPIC)
    submitted = st.form_submit_button("Générer les RQ et les concepts (TD02)", type="primary")
if submitted:
    for key in ("td02", "validated", "runs", "evaluation"):
        st.session_state.pop(key, None)
    rq_model, concept_model, _, _ = attempt("Construction du LLM", models)
    outcome = attempt("Génération des RQ", lambda: generate_questions_for_topic(rq_model, topic))
    proposals = []
    if outcome.status != "human_required":
        proposals = attempt("Extraction des concepts",
                            lambda: propose_concepts_for_topic(concept_model, outcome, topic))
    st.session_state["td02"] = {"topic": topic, "outcome": outcome, "proposals": proposals}

td02 = st.session_state.get("td02")
if td02 is None:
    st.info("Saisissez un sujet puis lancez le TD02 : ses concepts validés deviendront les concepts requis.")
    st.stop()
if llm_mode == "demo" and td02["topic"] != REFERENCE_TOPIC:
    st.warning("Le LLM de démonstration rejoue les réponses du sujet de référence.")
outcome = td02["outcome"]
if outcome.status == "human_required":
    st.error(f"RQ non validées : décision humaine requise ({outcome.error}).")
    st.stop()
columns = st.columns(len(td02["proposals"]))
edited = {}
for column, proposal in zip(columns, td02["proposals"]):
    with column:
        default = format_concepts_text(proposal.extraction.concepts) if proposal.extraction else ""
        edited[proposal.identifier] = st.text_area(
            f"{proposal.identifier} · {proposal.question}", value=default, height=130,
            key=f"concepts_{proposal.identifier}_{hash(td02['topic'])}")
if st.button("Valider les concepts (décision humaine)"):
    try:
        st.session_state["validated"] = [apply_human_review(p, parse_concepts_text(edited[p.identifier]))
                                         for p in td02["proposals"]]
        st.session_state.pop("runs", None)
    except ValueError as error:
        st.error(f"Décision refusée par le contrat Pydantic : {short_error(error)}")
validated = st.session_state.get("validated")
if validated is None:
    st.info("Relisez, modifiez si besoin, puis validez les concepts.")
    st.stop()

# ---------------------------------------------------------------------------
# 2. Corpus et outils (acquis TD05)
# ---------------------------------------------------------------------------
st.header("2. Corpus et boîte à outils (acquis du TD05)")
documents, lineage = attempt("Chargement des artefacts", load_rag_handoff)
chunks = build_corpus(documents)
embeddings = DemoEmbeddings() if embedding_mode == "demo" else build_embeddings(Settings())
retrievers = attempt("Indexation offline", lambda: make_retrievers(chunks, embeddings))
st.caption(f"Corpus RAG : {lineage['rag_documents']} articles (TD03 ∩ TD04 ∩ disponibles), {len(chunks)} chunks · "
           f"outils de l'agent : retrievers {', '.join(retrievers)}, extracteur structuré, décomposeur LLM.")


def toolbox_for(memory):
    _, _, extraction_model, _ = models()
    return Toolbox(retrievers=retrievers, extractor=build_llm_extractor(extraction_model),
                   decomposer=make_decomposer(extraction_model), memory=memory)


def run_config(config, question, concepts, memory):
    if config == "naive":
        return run_fixed(question, concepts, toolbox_for(None), NAIVE, config)
    if config == "advanced":
        return run_fixed(question, concepts, toolbox_for(None), ADVANCED, config)
    if config == "agentique-llm":
        decide = make_llm_decider(models()[3], list(retrievers))
        return run_agentic(question, concepts, toolbox_for(None), decide, config=config)
    return run_agentic(question, concepts, toolbox_for(memory if config == "agentique+memoire" else None),
                       config=config)


configs = ["naive", "advanced", "agentique", "agentique+memoire"] + (["agentique-llm"]
                                                                     if decider_mode == "llm" else [])

# ---------------------------------------------------------------------------
# 3. Une question, plusieurs configurations
# ---------------------------------------------------------------------------
st.header("3. Une question : naïf, avancé, agentique")
eval_set = {item.id: item for item in load_eval_set()}
items = {item.identifier: item for item in validated}
options = list(items) + list(eval_set)
labels = {**{k: f"{k} (TD02) — {v.question}" for k, v in items.items()},
          **{k: f"{k} [{v.kind}] — {v.question}" for k, v in eval_set.items()}}
choice = st.selectbox("Question à documenter", options, index=options.index("E3"), format_func=labels.get)
if choice in items:
    question = items[choice].question
    concepts = concepts_from_td02(items[choice], [v for k, v in items.items() if k != choice])
else:
    question, concepts = eval_set[choice].question, None
required = concepts or concepts_from_question(question)
st.caption("Concepts requis : " + ", ".join(c.label for c in required)
           + (" (concepts propres à la RQ, validés au TD02)" if concepts else " (vocabulaire contrôlé)"))

if st.button("Comparer les configurations", type="primary", icon="⚖️"):
    memory = load_memory()
    runs, missing = [], {}
    for config in configs:
        try:
            runs.append(run_config(config, question, concepts, memory))
        except NotImplementedError as error:
            missing[config] = str(error)
    if any(state.config == "agentique+memoire" for state in runs):
        save_memory(memory)
        save_agentic_evidence(next(s for s in runs if s.config == "agentique+memoire"), td02["topic"])
    st.session_state["runs"] = {"choice": choice, "states": runs, "missing": missing}

result = st.session_state.get("runs")
if result is None or result["choice"] != choice:
    st.info("Lancez la comparaison pour cette question.")
    st.stop()
for config, error in result["missing"].items():
    st.warning(f"**{CONFIG_LABELS[config]}** non disponible : {error}.")
annotation = annotation_for(question)
rows = []
for state in result["states"]:
    row = {"configuration": CONFIG_LABELS[state.config], "issue": state.outcome, "pas": state.steps,
           "appels LLM": state.llm_calls, "appels retriever": state.retriever_calls,
           "preuves": " · ".join(f"{f.evidence.status} p.{f.evidence.page}" for f in state.focuses if f.evidence)}
    if annotation:
        try:
            row["correct"] = "✓" if evaluate_run(state, annotation)["correct"] else "✗"
        except NotImplementedError:
            row["correct"] = "TODO 10"
    rows.append(row)
st.dataframe(rows, hide_index=True)
if result["states"]:
    tabs = st.tabs([CONFIG_LABELS[state.config] for state in result["states"]])
    for tab, state in zip(tabs, result["states"]):
        with tab:
            st.dataframe([{"pas": t.step, "statut": STATUS_ICONS[t.status] + " " + t.status,
                           "action": (t.action or {}).get("action", "—"),
                           "raison": (t.action or {}).get("reason", ""), "observation": t.observation}
                          for t in state.trace], hide_index=True,
                         column_config={"observation": st.column_config.TextColumn(width="large")})
            for index, focus in enumerate(state.focuses):
                if focus.evidence:
                    message = (f"[{index}] {focus.question} → « {focus.evidence.supporting_quote} » "
                               f"({focus.evidence.chunk_id}, p. {focus.evidence.page})")
                    {"verified": st.success, "human_review": st.warning,
                     "rejected": st.error}[focus.evidence.status](message)
            if state.outcome == "abstention":
                st.info("Abstention : aucune preuve suffisante n'a été trouvée ; aucune réponse n'est produite.")
if any(state.config == "agentique+memoire" for state in result["states"]):
    st.success("Preuves de la configuration « agentique + mémoire » enregistrées : artifacts/verified_evidence.json")

# ---------------------------------------------------------------------------
# 4. Mémoire épisodique
# ---------------------------------------------------------------------------
st.header("4. Mémoire épisodique")
memory = load_memory()
if memory.episodes:
    st.dataframe([{"concepts": ", ".join(e.concepts), "stratégie gagnante": e.strategy.label(),
                   "utilisations": e.uses, "question d'origine": e.question} for e in memory.episodes],
                 hide_index=True)
else:
    st.caption("Mémoire vide : seules les preuves VERIFIED produisent un souvenir.")
if st.button("Oublier (mémoire vide)"):
    save_memory(AgentMemory())
    st.rerun()

# ---------------------------------------------------------------------------
# 5. Évaluation expérimentale
# ---------------------------------------------------------------------------
st.header("5. Évaluation : même corpus, mêmes questions, même contrôle")
st.caption("Le protocole repart d'une mémoire vide ; les questions sont traitées dans l'ordre E1 → E8.")
if st.button("Lancer l'évaluation", icon="📊"):
    fresh, rows = AgentMemory(), []
    for item in eval_set.values():
        for config in configs:
            try:
                state = run_config(config, item.question, None, fresh)
            except NotImplementedError:
                continue
            metrics = attempt("Évaluation (TODO 10)", lambda: evaluate_run(state, item))
            rows.append({"id": item.id, "kind": item.kind, "config": state.config, **metrics})
    summary = summarize(rows)
    save_report(rows, summary)
    st.session_state["evaluation"] = {"rows": rows, "summary": summary}
evaluation = st.session_state.get("evaluation")
if evaluation:
    st.dataframe([{"configuration": CONFIG_LABELS[r["config"]], "correctes": f"{r['correctes']}/{r['questions']}",
                   "fidélité": r["fidelite"], "pertinence": r["pertinence"], "appels LLM": r["appels_llm"],
                   "appels retriever": r["appels_retriever"], "pas": r["pas"]} for r in evaluation["summary"]],
                 hide_index=True)
    detail = {}
    for row in evaluation["rows"]:
        detail.setdefault(row["id"], {"question": row["id"], "type": row["kind"]})[
            CONFIG_LABELS[row["config"]]] = "✓" if row["correct"] else "✗"
    st.dataframe(list(detail.values()), hide_index=True)
    st.success("Rapport enregistré : artifacts/rag_comparison.json")
