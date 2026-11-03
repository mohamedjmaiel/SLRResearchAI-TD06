# Construire SLRResearchAI de bout en bout jusqu'au RAG agentique

## 1. Finalité du guide

Les TD05 et TD06 (RAG) fonctionnent seuls grâce à des artefacts préremplis. Dans le projet final, ces
fichiers ne sont plus des données isolées : ils sont produits successivement par les TD précédents.

```text
sujet scientifique
→ questions et requêtes (TD02)
→ recherche d'articles (TD03)
→ screening (TD04)
→ préparation des documents
→ RAG classique et preuves vérifiées (TD05)
→ RAG agentique, mémoire et évaluation (TD06)
→ rédaction scientifique
```

Deux modes de travail :

- **mode TD autonome** : utiliser les artefacts fournis pour se concentrer sur le RAG ;
- **mode projet intégré** : remplacer chaque artefact fourni par la sortie réelle du composant précédent.

## 2. Un seul projet Python qui grandit

```text
slrresearch/
├── research_questions.py, search_plan.py, td02_bridge.py   # TD01–TD02
├── tools.py, search_agent.py, models.py                    # TD03
├── abstracts.py, screening_agent.py                        # TD04
├── rag_models.py, rag_artifacts.py                         # TD05 : contrats, transmission
├── rag_index.py                                            # TD05 : indexation offline, BM25, hybride
├── rag_pipeline.py, pdf_loader.py                          # TD05 : pipeline, extraction PDF
├── agentic_models.py, agentic_tools.py                     # TD06 : contrats, outils, garde-fous
├── agentic_rag.py, agentic_artifacts.py                    # TD06 : agent RAG, mémoire, évaluation
├── demo_llm.py, demo_embeddings.py                         # mode démonstration
└── config.py                                               # configuration partagée (.env)
```

Deux mécanismes assurent la continuité :

1. les **imports Python** permettent aux nouveaux modules de réutiliser les classes existantes ;
2. les **artefacts JSON** conservent les sorties entre deux exécutions et rendent les étapes
   découplées et reproductibles.

## 3. Contrats entre les étapes

| Producteur | Artefact | Consommateur | Information transmise |
|---|---|---|---|
| TD02 | RQ + concepts validés (en mémoire) | TD03, TD05 | requêtes par moteur ; requête de retrieval |
| TD03 | `search_results.json` | TD04 | articles trouvés et dédupliqués |
| TD04 | `screening_decisions.json` | préparation documentaire | décision finale par article |
| Préparation documentaire | `document_manifest.json` | TD05 RAG | disponibilité et emplacement des documents |
| Extraction PDF | `extracted_pages.json` | TD05 RAG | texte conservé page par page |
| Annotation humaine | `rag_evaluation.json` | évaluation du retrieval | pages de référence par question |
| TD05 RAG | `verified_evidence.json` | TD de rédaction | preuves, citations et provenance |
| Annotation humaine | `data/eval_td06.json` | TD06 (évaluation) | pages acceptables et aspects par question |
| TD06 RAG agentique | `verified_evidence.json` (une preuve par sous-question) | TD de rédaction | preuves, stratégie gagnante |
| TD06 RAG agentique | `agent_memory.json` | exécutions suivantes | concepts → stratégie gagnante (preuves vérifiées seulement) |
| TD06 RAG agentique | `rag_comparison.json` | QualityAgent, rapport | qualité et coût par configuration |

L'identifiant `paper_id` est la clé de transmission commune : il n'est jamais recréé ni remplacé.

## 4. Mode TD autonome

```bash
python -m pip install -r requirements.txt
python -m pytest -v
python main.py
python -m streamlit run streamlit_app.py
```

Les artefacts de `artifacts/` permettent de terminer les TODO sans relancer les API
scientifiques, le screening ou le téléchargement. Ils ont été produits par la vraie chaîne
TD02 → TD03 → TD04 (mode démonstration) sur le sujet de référence.

## 5. La jointure d'entrée

```python
from slrresearch.rag_artifacts import load_rag_handoff

documents, lineage = load_rag_handoff()
```

`load_rag_handoff()` calcule : identifiants trouvés au TD03 ∩ inclus au TD04 ∩ dont le document
est disponible. Un article exclu, en attente d'arbitrage ou indisponible n'entre pas
silencieusement dans le corpus. `lineage["excluded_from_rag"]` indique pourquoi chaque article est écarté.

## 6. Remplacer les artefacts amont

- **TD03** : exporter le `SearchState` en `search_results.json` (`schema_version`, `origin = "TD03"`,
  `topic`, `rq`, `queries`, `papers`), sans renommer les identifiants.
- **TD04** : après screening et arbitrage humain, exporter une décision **finale** par article
  (`include` / `exclude`, jamais `uncertain`), avec son décideur et sa raison.
- **Documents** : pour chaque article inclus, récupérer légalement le PDF, renseigner son statut
  (`downloaded`, `manual_required`, `unavailable`) et extraire les pages avec
  `extract_pdf_pages("documents/W1001.pdf")` ; un PDF scanné exige un OCR.

## 7. Construire puis interroger le corpus

```python
from slrresearch.demo_embeddings import DemoEmbeddings
from slrresearch.rag_pipeline import build_corpus, make_retrievers

chunks = build_corpus(documents, size=80, overlap=20)       # paper_id + chunk_id + page + text
retrievers = make_retrievers(chunks, DemoEmbeddings())      # OFFLINE : index construits une fois
results = retrievers["semantique"]("recall sensitivity screening", 3)   # ONLINE : consultation
```

Comparer les retrievers exige le même corpus, la même requête et le même `k`, puis Precision@k et
Recall@k sur des pages de référence annotées par un humain (`rag_evaluation.json`).

## 8. Extraire, vérifier, transmettre

```python
from slrresearch.rag_pipeline import build_llm_extractor, run_rag
from slrresearch.rag_artifacts import save_evidence_artifact

run = run_rag(question, retrieval_query, retrievers["semantique"], build_llm_extractor(model), k=3)
save_evidence_artifact(run, topic, "semantique")            # artifacts/verified_evidence.json
```

Le LLM propose une réponse et une citation ; le programme vérifie le chunk transmis, l'article,
la page et la présence de la citation. Seules les preuves `verified` sont
`authorized_for_writing` : le futur WritingAgent ne citera que celles-ci.

## 9. Vérifications de bout en bout

- [ ] les `paper_id` du screening existent dans les résultats de recherche ;
- [ ] chaque papier inclus possède une décision finale ;
- [ ] les documents indisponibles ne sont pas indexés ;
- [ ] chaque chunk conserve article, page et identifiant ;
- [ ] le corpus est indexé une fois, puis seulement consulté ;
- [ ] les retrievers comparés utilisent le même corpus, la même requête et le même `k` ;
- [ ] les métriques ne sont calculées qu'avec des références annotées ;
- [ ] chaque preuve est contrôlée après la génération ; les cas `human_review` restent visibles ;
- [ ] `verified_evidence.json` est produit.

## 10. Du pipeline fixe à l'agent (TD06)

```python
from slrresearch.agentic_rag import ADVANCED, NAIVE, run_agentic, run_fixed, evaluate_run
from slrresearch.agentic_tools import Toolbox, make_decomposer
from slrresearch.agentic_artifacts import load_memory, save_memory

memory = load_memory()                                     # artifacts/agent_memory.json
toolbox = Toolbox(retrievers=retrievers, extractor=build_llm_extractor(model),
                  decomposer=make_decomposer(model), memory=memory)
naive = run_fixed(question, None, toolbox, NAIVE, "naive")  # TD05 : stratégie fixe
agent = run_agentic(question, None, toolbox)                # boucle bornée + mémorisation
save_memory(memory)
```

- [ ] les concepts requis d'une question sont explicites (vocabulaire contrôlé ou TD02) ;
- [ ] toute action de l'agent passe par `check_action` ; les refus sont tracés ;
- [ ] la boucle est bornée (pas, recherches, k) et se termine par une réponse ou une abstention ;
- [ ] seules les preuves `verified` alimentent la mémoire et la rédaction ;
- [ ] les configurations sont comparées sur le même corpus, les mêmes questions et le même contrôle,
      avec leur coût.

