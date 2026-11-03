"""TD06 - RAG agentique : du sujet saisi à la comparaison expérimentale (ligne de commande).

    python main.py                                  # mode guidé : sujet, TD02, question, évaluation
    python main.py --sujet "..." --rq RQ1           # une RQ du TD02 (concepts acceptés)
    python main.py --sujet "..." --question E3      # une question du jeu d'évaluation (E1..E8)
    python main.py --sujet "..." --evaluation       # tout le jeu d'évaluation, 4 configurations
    python main.py --decideur llm --scenario indiscipline   # le LLM décide (niveau 3)
    python main.py --llm reel --embeddings ollama   # LLM du .env et embeddings Ollama

Chaîne exécutée :
    1. sujet → TD02 : RQ validées, concepts (revue humaine)
    2. corpus du TD05 : TD03 ∩ TD04 ∩ documents, chunks, index, retrievers (acquis)
    3. une question → RAG naïf / avancé / agentique, trace de l'agent       TODO 1-7
    4. mémoire épisodique (artifacts/agent_memory.json)                     TODO 8-9
    5. évaluation : même corpus, mêmes questions, même contrôle             TODO 10
"""
import argparse

from slrresearch.agentic_artifacts import (
    MEMORY_PATH,
    annotation_for,
    load_eval_set,
    load_memory,
    save_agentic_evidence,
    save_memory,
    save_report,
)
from slrresearch.agentic_models import AgentMemory, AgentState
from slrresearch.agentic_rag import (
    ADVANCED,
    NAIVE,
    evaluate_run,
    rewrite_query,
    run_agentic,
    run_fixed,
    summarize,
)
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
from slrresearch.search_plan import (
    ConceptProposal,
    SearchConcepts,
    accept_proposal,
    apply_human_review,
    format_concepts_text,
    parse_concepts_text,
)
from slrresearch.td02_bridge import (
    CONCEPT_MAX_OUTPUT_TOKENS,
    REFERENCE_TOPIC,
    generate_questions_for_topic,
    propose_concepts_for_topic,
)

EXTRACTION_MAX_OUTPUT_TOKENS = 800
DECISION_MAX_OUTPUT_TOKENS = 300
CONFIG_LABELS = {"naive": "RAG naïf (TD05, question brute)", "advanced": "RAG avancé (reformulation + hybride)",
                 "agentique": "RAG agentique (politique codée)", "agentique+memoire": "RAG agentique + mémoire",
                 "agentique-llm": "RAG agentique (décideur LLM)"}


class Stop(Exception):
    """Arrêt propre de la chaîne (TODO non réalisé, LLM indisponible...)."""


def parse_args():
    parser = argparse.ArgumentParser(description="TD06 - RAG agentique")
    parser.add_argument("--sujet", help="sujet de la revue (sinon saisi au clavier)")
    parser.add_argument("--rq", help="RQ du TD02 à documenter (ex. RQ1)")
    parser.add_argument("--question", help="question du jeu d'évaluation (E1..E8) ou texte libre")
    parser.add_argument("--evaluation", action="store_true", help="évaluer tout le jeu d'évaluation")
    parser.add_argument("--accepter", action="store_true", help="accepter les concepts proposés (TD02)")
    parser.add_argument("--decideur", choices=["regles", "llm"], default="regles",
                        help="politique codée (niveau 2) ou décideur LLM (niveau 3)")
    parser.add_argument("--scenario", choices=REACT_SCENARIOS, default="discipline",
                        help="scénario du décideur LLM de démonstration")
    parser.add_argument("--llm", choices=["demo", "reel"], default="demo", help="LLM de démonstration ou .env")
    parser.add_argument("--embeddings", choices=["demo", "ollama"], default="demo",
                        help="embeddings de démonstration ou nomic-embed-text (Ollama)")
    parser.add_argument("--comportement", choices=RAG_BEHAVIOURS, default="fidele",
                        help="comportement du LLM de démonstration lors de l'extraction")
    parser.add_argument("--oublier", action="store_true", help="repartir d'une mémoire épisodique vide")
    return parser.parse_args()


def read(prompt: str, default: str = "") -> str:
    """input() avec valeur par défaut (Entrée, ou entrée standard fermée)."""
    try:
        return input(prompt).strip() or default
    except EOFError:
        print()
        return default


def guard(step: str, action):
    """Exécute une étape ; un TODO manquant ou un service indisponible arrête proprement."""
    try:
        return action()
    except NotImplementedError as error:
        raise Stop(f"{step} non disponible : {error}. Complétez ce TODO puis relancez.") from error
    except Exception as error:  # clé absente, Ollama arrêté, quota, contrat violé...
        raise Stop(f"{step} impossible : {short_error(error)}") from error


def build_models(args):
    """(modèle des RQ, modèle des concepts, modèle d'extraction et de décomposition, modèle de décision)."""
    if args.llm == "demo":
        model = DemoChatModel(react_scenario=args.scenario, rag_behaviour=args.comportement)
        return model, model, model, model
    settings = Settings()
    name = settings.ollama_model if settings.provider == "ollama" else settings.groq_model
    print(f"LLM réel : {settings.provider} | {name}")
    return (build_model(settings, resolve_decoding("greedy")),
            build_model(settings, resolve_decoding("greedy", max_output_tokens=CONCEPT_MAX_OUTPUT_TOKENS)),
            build_model(settings, resolve_decoding("greedy", max_output_tokens=EXTRACTION_MAX_OUTPUT_TOKENS)),
            build_model(settings, resolve_decoding("greedy", max_output_tokens=DECISION_MAX_OUTPUT_TOKENS)))


def review_proposal(proposal: ConceptProposal, accept: bool) -> SearchConcepts:
    """Revue humaine du TD02 : les concepts validés deviendront les concepts requis."""
    print(f"\n--- {proposal.identifier} · {proposal.question}")
    if proposal.extraction:
        print(format_concepts_text(proposal.extraction.concepts))
        if accept or read("Accepter ces concepts ? [O/n] : ", "o").lower() in ("o", "oui", "y"):
            return accept_proposal(proposal)
    while True:
        print("Saisissez un concept par ligne « concept: terme1; terme2 » (ligne vide = fin).")
        lines = []
        while line := read("  > "):
            lines.append(line)
        if not lines and proposal.extraction:
            print("Aucune saisie : les concepts proposés sont conservés.")
            return accept_proposal(proposal)
        try:
            return apply_human_review(proposal, parse_concepts_text("\n".join(lines)))
        except ValueError as error:
            print(f"Saisie refusée par le contrat Pydantic : {error}")


def print_state(state: AgentState) -> None:
    print(f"\n  {CONFIG_LABELS.get(state.config, state.config)}")
    for step in state.trace:
        action = (step.action or {}).get("action", "?")
        print(f"    {step.step:>2}. [{step.status:8}] {action:9} {step.observation}")
        reason = (step.action or {}).get("reason")
        if reason and state.config.startswith("agentique"):
            print(f"        raison : {reason}")
    for index, focus in enumerate(state.focuses):
        evidence = focus.evidence
        if evidence:
            print(f"    preuve [{index}] {evidence.status.upper():12} {evidence.chunk_id} p. {evidence.page} "
                  f"« {evidence.supporting_quote[:80]} »")
    print(f"    issue : {state.outcome} · appels LLM {state.llm_calls} · appels retriever "
          f"{state.retriever_calls} · pas {state.steps}")


def run_configurations(question, concepts, toolbox_for, decide, memory, configs) -> list[AgentState]:
    """Exécute chaque configuration ; un TODO manquant n'empêche pas les configurations suivantes."""
    states = []
    for config in configs:
        try:
            if config == "naive":
                state = run_fixed(question, concepts, toolbox_for(None), NAIVE, config)
            elif config == "advanced":
                state = run_fixed(question, concepts, toolbox_for(None), ADVANCED, config)
            elif config == "agentique+memoire":
                state = run_agentic(question, concepts, toolbox_for(memory), config=config)
            elif config == "agentique-llm":
                state = run_agentic(question, concepts, toolbox_for(None), decide, config=config)
            else:
                state = run_agentic(question, concepts, toolbox_for(None), config=config)
        except NotImplementedError as error:
            print(f"\n  {CONFIG_LABELS[config]}\n    non disponible : {error}")
            continue
        states.append(state)
    return states


def run(args) -> None:
    guided = args.sujet is None
    model, concept_model, extraction_model, decision_model = guard("Construction du LLM",
                                                                   lambda: build_models(args))

    # 1. Sujet → TD02 --------------------------------------------------------
    topic = args.sujet or read(f"Sujet de la revue [{REFERENCE_TOPIC}] : ", REFERENCE_TOPIC)
    print(f"\n=== 1. Sujet → TD02 ===\nSujet : {topic}")
    if args.llm == "demo" and topic != REFERENCE_TOPIC:
        print("Attention : le LLM de démonstration rejoue les réponses du sujet de référence.")
    outcome = guard("Génération des RQ (TD02)", lambda: generate_questions_for_topic(model, topic))
    if outcome.status == "human_required":
        raise Stop(f"Validation humaine requise ({outcome.error}).")
    for question in outcome.questions:
        print(f"  {question.identifier}. {question.question}")
    proposals = guard("Extraction des concepts (TD02)",
                      lambda: propose_concepts_for_topic(concept_model, outcome, topic))
    validated = [review_proposal(p, args.accepter or not guided) for p in proposals]
    items = {item.identifier: item for item in validated}

    # 2. Corpus et retrievers (acquis TD05) ------------------------------------
    print("\n=== 2. Corpus du TD05 : TD03 ∩ TD04 ∩ documents, chunks, index ===")
    documents, lineage = guard("Chargement des artefacts", load_rag_handoff)
    chunks = build_corpus(documents)
    embeddings = DemoEmbeddings() if args.embeddings == "demo" else build_embeddings(Settings())
    retrievers = guard("Indexation offline", lambda: make_retrievers(chunks, embeddings))
    print(f"  corpus RAG : {lineage['rag_documents']} articles, {len(chunks)} chunks, retrievers {list(retrievers)}")
    extractor = build_llm_extractor(extraction_model)
    decomposer = make_decomposer(extraction_model)
    decide = make_llm_decider(decision_model, list(retrievers)) if args.decideur == "llm" else None

    def toolbox_for(memory):
        return Toolbox(retrievers=retrievers, extractor=extractor, decomposer=decomposer, memory=memory)

    configs = ["naive", "advanced", "agentique", "agentique+memoire"]
    if args.decideur == "llm":
        configs.append("agentique-llm")

    # 3. Une question ----------------------------------------------------------
    eval_set = {item.id: item for item in load_eval_set()}
    choice = args.question or args.rq
    if choice is None and guided:
        print("\nQuestions disponibles : " + ", ".join(items) + " (TD02) ou "
              + ", ".join(eval_set) + " (jeu d'évaluation)")
        for key, item in eval_set.items():
            print(f"  {key} [{item.kind}] {item.question}")
        choice = read("Question à documenter [E3] : ", "E3")
    choice = choice or "E3"
    if choice in items:
        others = [item for key, item in items.items() if key != choice]
        question, concepts = items[choice].question, concepts_from_td02(items[choice], others)
        origin = "concepts validés au TD02, propres à cette RQ"
    elif choice in eval_set:
        question, concepts = eval_set[choice].question, None
        origin = "vocabulaire contrôlé (GLOSSARY)"
    else:
        question, concepts, origin = choice, None, "vocabulaire contrôlé (GLOSSARY)"
    required = concepts or concepts_from_question(question)
    print(f"\n=== 3. Une question, plusieurs configurations — TODO 1-7 ===\nQuestion : {question}")
    if not required:
        raise Stop("Aucun concept requis détecté : choisissez une RQ du TD02 ou complétez GLOSSARY.")
    print(f"  concepts requis ({origin}) : " + ", ".join(c.label for c in required))
    try:
        print(f"  requête reformulée (TODO 3) : {rewrite_query(required)}")
    except NotImplementedError as error:
        print(f"  requête reformulée non disponible : {error}")

    memory = AgentMemory() if args.oublier else load_memory()
    print(f"  mémoire épisodique : {len(memory.episodes)} souvenir(s) ({MEMORY_PATH.name})")
    states = run_configurations(question, concepts, toolbox_for, decide, memory, configs)
    annotation = annotation_for(question)
    for state in states:
        print_state(state)
        if annotation:
            try:
                metrics = evaluate_run(state, annotation)
                print(f"    évaluation : {'CORRECT' if metrics['correct'] else 'INCORRECT'} "
                      f"(fidélité {metrics['fidelite']}, pertinence {metrics['pertinence']})")
            except NotImplementedError as error:
                print(f"    évaluation non disponible : {error}")
    agentic = next((s for s in states if s.config == "agentique+memoire"), None)
    if agentic is not None:
        save_memory(memory)
        print(f"\n=== 4. Mémoire épisodique — TODO 8-9 ===\n  {len(memory.episodes)} souvenir(s) enregistrés :")
        for episode in memory.episodes:
            print(f"    {episode.concepts} → {episode.strategy.label()} (utilisé {episode.uses} fois)")
        records = save_agentic_evidence(agentic, topic)
        authorized = sum(r["authorized_for_writing"] for r in records)
        print(f"  Artefact : artifacts/verified_evidence.json ({len(records)} preuve(s), "
              f"{authorized} autorisée(s) pour la rédaction)")

    # 5. Évaluation expérimentale --------------------------------------------
    evaluate = args.evaluation or (guided and read("\nLancer l'évaluation sur le jeu complet ? [O/n] : ",
                                                   "o").lower() in ("o", "oui", "y"))
    if not evaluate:
        return
    print("\n=== 5. Évaluation : même corpus, mêmes questions, même contrôle — TODO 10 ===")
    fresh = AgentMemory()                        # protocole reproductible : mémoire vide au départ
    rows = []
    for item in eval_set.values():
        for state in run_configurations(item.question, None, toolbox_for, decide, fresh, configs):
            metrics = guard("Évaluation (TODO 10)", lambda: evaluate_run(state, item))
            rows.append({"id": item.id, "kind": item.kind, "config": state.config, **metrics})
    summary = summarize(rows)
    header = f"  {'configuration':38} correctes  fidélité  pertinence  LLM   retriever  pas"
    print(header)
    for row in summary:
        print(f"  {CONFIG_LABELS[row['config']]:38} {row['correctes']:>3}/{row['questions']:<4}"
              f"  {str(row['fidelite']):>8}  {str(row['pertinence']):>10}  {row['appels_llm']:>4}"
              f"  {row['appels_retriever']:>9}  {row['pas']:>4}")
    print("\n  Détail (✓ = correct) :")
    for item in eval_set.values():
        marks = " ".join(f"{r['config']}={'✓' if r['correct'] else '✗'}" for r in rows if r["id"] == item.id)
        print(f"    {item.id} [{item.kind:13}] {marks}")
    save_report(rows, summary)
    print("  Rapport : artifacts/rag_comparison.json")


def main() -> None:
    try:
        run(parse_args())
    except Stop as stop:
        print(f"\n{stop}")


if __name__ == "__main__":
    main()
