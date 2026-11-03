# TD06 RAG agentique — Décider, chercher autrement, s'abstenir, mémoriser, mesurer

**Séance 07 · Chapitre RAG, partie II · Durée : 90 minutes · Projet fil rouge SLRResearchAI**

## 1. Objectif

Au TD05, le pipeline RAG était **fixe** : une requête, un retriever, un Top-k, une génération,
un contrôle. Ce TD transforme ce pipeline en **agent** : à chaque étape, un décideur choisit la
prochaine action à partir de ce qu'il **observe**, dans une boucle **bornée** et **contrôlée**
par le logiciel.

```text
RAG naïf      question ─→ retriever fixe ─→ Top-k ─→ LLM ─→ contrôle                 (TD05)
RAG avancé    question ─→ reformulation ─→ retriever hybride ─→ Top-k ─→ LLM ─→ contrôle
RAG agentique ┌──────────────────────────────────────────────────────────────┐
              │ décider ─→ contrôler l'action ─→ agir ─→ OBSERVER (preuves ?) │  ≤ 10 pas
              └──────────────────────────────────────────────────────────────┘
              actions : chercher (stratégie) · décomposer · répondre · s'abstenir
              mémoire épisodique : « pour ces concepts, cette stratégie a marché »
```

Vous comparerez ensuite les trois architectures sur le **même corpus**, les **mêmes questions**
annotées et le **même contrôle** des preuves : réponses correctes, fidélité, pertinence, coût
(appels LLM et retriever) et latence (nombre de pas).

**À la fin de la séance, vous saurez :**

1. expliquer trois limites du RAG classique (vocabulaire, question à plusieurs aspects, absence de réponse dans le corpus) et les observer ;
2. distinguer RAG naïf, avancé (pré- et post-retrieval) et agentique (le chemin est décidé pendant l'exécution) ;
3. transformer une observation (« les preuves suffisent-elles ? ») en décision : reformuler, changer de retriever, élargir k, décomposer, répondre ou s'abstenir ;
4. borner et contrôler un agent RAG : actions autorisées, budget, pas de répétition, refus tracés ;
5. écrire et relire une mémoire épisodique sans l'empoisonner ;
6. évaluer plusieurs configurations selon un protocole équitable, en rapportant qualité **et** coût.

**Prérequis :** TD03 (boucle agentique, politique codée, décideur LLM contrôlé), TD05 (retrievers,
contexte, extraction structurée, `verify_evidence`), cours de la séance 07. Aucune clé ni réseau :
un LLM et des embeddings de démonstration sont fournis.

## 2. Continuité avec les TD précédents

Le TD06 est une **extension du même projet Python**. Le pipeline du TD05 (version corrigée,
`rag_pipeline.py`) devient une **boîte à outils** de l'agent : les quatre retrievers, l'extracteur
structuré et la vérification des preuves sont réutilisés tels quels.

| Acquis | Rôle dans le TD06 |
|---|---|
| TD02 : RQ et concepts validés par l'humain | concepts requis d'une RQ (`concepts_from_td02`) |
| TD03 : boucle « décider → contrôler → agir → observer », politique codée, décideur LLM | même architecture, appliquée au retrieval |
| TD05 : chunks, index, retrievers, `run_rag`, `verify_evidence` | outils de l'agent ; RAG naïf et avancé (`run_fixed`) |
| TD06 : nouvelle capacité | `agentic_models.py`, `agentic_tools.py`, `agentic_artifacts.py`, `agentic_rag.py` |

Les tests de `tests/acquis_td05/` et `tests/acquis_td04/` sont des **tests de non-régression** :
ils doivent rester verts.

**Même corpus, même transmission.** Le corpus est celui du TD05 : articles trouvés (TD03) ∩ inclus
(TD04) ∩ disponibles, soit 2 articles, 10 pages, 15 chunks. Les artefacts amont sont inchangés.

```text
artifacts TD03 ∩ TD04 ∩ documents ─→ corpus (15 chunks) ─→ retrievers (TD05)
data/eval_td06.json (8 questions annotées) ─→ agent ─→ artifacts/verified_evidence.json
                                                     ─→ artifacts/agent_memory.json
                                                     ─→ artifacts/rag_comparison.json
```

**Le sujet est saisi par l'utilisateur.** `main.py` et `streamlit_app.py` exécutent le TD02 comme
au TD05. Vous pouvez documenter une RQ du TD02 (ses concepts validés deviennent les concepts
requis) ou une question du **jeu d'évaluation** `data/eval_td06.json` (annotée par deux relecteurs).

## 3. Fichiers et architecture

- `slrresearch/agentic_models.py` : **contrats fournis, à lire** (`Concept`, `Strategy`,
  `Assessment`, `Focus`, `AgentAction`, `AgentState`, `Episode`, `AgentMemory`, `EvalQuestion`) ;
- `slrresearch/agentic_tools.py` : **fourni, à lire** : vocabulaire contrôlé `GLOSSARY`,
  `contains_term`, échelle `STRATEGIES`, budgets `MAX_STEPS`, `MAX_RETRIEVALS`, `MAX_K`,
  `Toolbox`, décomposeur LLM, décideur LLM (niveau 3) ;
- `slrresearch/agentic_artifacts.py` : jeu d'évaluation, mémoire persistée, rapport, preuves ;
- `slrresearch/agentic_rag.py` : **seul fichier contenant les TODO** ; son code « fourni »
  (`execute`, `finalize`, `run_fixed`, `run_agentic`, `summarize`) est à lire ;
- `data/eval_td06.json` : 8 questions annotées (pages acceptables, nombre d'aspects) ;
- `tests/test_td06.py` : un groupe de tests indépendant par TODO + tests « fourni » ;
- `main.py`, `streamlit_app.py` : saisie du sujet, comparaison des configurations, trace de
  l'agent, mémoire, évaluation.

**Notions mobilisées (rappels du cours).**

- *Concepts requis* : ce qu'un passage doit contenir pour répondre. Ils viennent du vocabulaire
  contrôlé de la revue (`GLOSSARY`, français → termes du corpus) ou des concepts validés au TD02.
  Seuls les concepts **discriminants** comptent : « LLM » ou « screening » sont partout dans ce corpus.
- *Observation* (`Assessment`) : le **meilleur passage** couvre-t-il tous les concepts ?
  `suffisante`, `partielle` ou `insuffisante`. C'est un juge à règles, explicable et reproductible ;
  en production, on utilise aussi un modèle de reranking ou un LLM juge.
- *Stratégie* = (retriever, type de requête, k). L'**échelle** `STRATEGIES` commence comme le RAG
  naïf, puis reformule, change de retriever, élargit k : on ne paie un effort supplémentaire que
  si l'observation le justifie (**récupération adaptative**).
- *Reformulation* (pré-retrieval, RAG avancé) : remplacer la question par les termes du corpus.
- *Décomposition* : une question à plusieurs aspects devient des sous-questions (**focus**), chacune
  avec ses passages et sa preuve.
- *Garde-fous* : le décideur (règles ou LLM) **propose** ; le logiciel **dispose** (`check_action`).
  Une action refusée devient une observation et consomme un pas.
- *Mémoire* : à **court terme**, l'`AgentState` d'une question (stratégies essayées, passages vus) ;
  à **long terme**, la mémoire **épisodique** (`agent_memory.json`) : concepts → stratégie gagnante.
  Un souvenir est une **suggestion**, jamais une preuve : la preuve est toujours recherchée et vérifiée.
- *Évaluation* : `correct` (réponse complète sur une page acceptable, ou abstention attendue),
  `fidelite` (preuves `verified` / preuves produites), `pertinence` (preuves vérifiées sur une page
  acceptable / preuves vérifiées), coût (appels LLM, appels retriever), latence (pas).

## 4. Installation et commandes

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -v
python main.py                     # saisissez le sujet, ou Entrée pour le sujet de référence
```

État initial attendu : **112 tests verts** (12 « fourni » + 100 de non-régression TD04/TD05) et
**35 tests rouges** (ceux des TODO). `python main.py` exécute le TD02, puis le **RAG naïf** sur la
question choisie (E3 par défaut) ; les autres configurations affichent le TODO qui manque
(`non disponible : TODO 3 - rewrite_query`…). Relancez-le après chaque TODO.

Test ciblé :

```bash
python -m pytest -v -k "todo_6_"
```

**Organisation.** Tout le travail demandé se fait en séance ; la difficulté augmente :
★ guidé (indice dans le code), ★★ semi-guidé (règles dans la docstring), ★★★ autonome (contrat seul).

| Minutes | Travail |
|---|---|
| 0 – 8 | installation, tests, `main.py` (RAG naïf sur E3 et E4), lecture des sections 2-3, `agentic_models.py`, `agentic_tools.py` |
| 8 – 20 | TODO 1 à 3 (observer, reformuler) |
| 20 – 30 | TODO 4 et 5 (agir, garde-fous) |
| 30 – 50 | TODO 6 et 7 (politique, boucle), puis traces E3, E4 et décideur LLM indiscipliné |
| 50 – 62 | TODO 8 et 9 (mémoire), puis E5 suivi de E7 |
| 62 – 74 | TODO 10 (évaluation), puis tableau comparatif |
| 74 – 80 | observations et synthèse |
| 80 – 90 | quiz de fin de séance (formulaire en ligne, individuel) |

**Travail en séance / travail à domicile.** Les 10 TODO, les tests et les observations se font
intégralement en séance. Aucun TODO n'est à terminer à domicile ; seuls des prolongements
facultatifs sont proposés (section 18).

## 5. TODO 1 — Repérer les concepts couverts ★

**Fichier :** `slrresearch/agentic_rag.py` — **fonction :** `covered_concepts(text, concepts)`

Retournez les libellés des concepts dont **au moins un** terme apparaît dans le texte, dans
l'ordre de la liste `concepts`. Utilisez `contains_term` (fourni) : il compare des mots ou
expressions **entiers**, sans tenir compte de la casse (« disagreement » ne contient pas « agreement »).

```bash
python -m pytest -v -k "todo_1_"
```

## 6. TODO 2 — Juger si les preuves suffisent ★★

**Fonction :** `assess_passages(passages, concepts)` → `Assessment`

C'est l'**observation** de l'agent après une recherche.

- liste de concepts vide : `ValueError` (on ne juge pas sans critère) ;
- meilleur passage = celui qui couvre le **plus** de concepts ; à égalité, le **premier** de la liste ;
- tous les concepts couverts → `suffisante` (`best_chunk_id`, `covered` = tous, `missing` = []) ;
- au moins un → `partielle` (`best_chunk_id`, `covered`, `missing`) ;
- aucun, ou aucun passage → `insuffisante` (`best_chunk_id = None`, `missing` = tous).

Pourquoi exige-t-on qu'**un seul** passage couvre tous les concepts, plutôt que l'ensemble des
passages ? (Indice : la preuve finale est **une** citation.)

```bash
python -m pytest -v -k "todo_2_"
```

## 7. TODO 3 — Reformuler la requête ★

**Fonction :** `rewrite_query(concepts)`

Reformulation pré-retrieval du RAG avancé : concaténez les termes du corpus de tous les concepts,
dans l'ordre, **sans doublon** (comparaison insensible à la casse ; on garde la première graphie),
séparés par une espace. Exemple : `[cout, vote_majoritaire]` → `"cost dollars majority voting"`.

```bash
python -m pytest -v -k "todo_3_"
python main.py --sujet "Utilisation des LLM pour assister le screening des SLR en génie logiciel" --question E3
```

Le **RAG avancé** fonctionne : comparez sa preuve à celle du RAG naïf. Laquelle répond à la
question entière ?

## 8. TODO 4 — Enregistrer le résultat d'une recherche ★★

**Fonction :** `record_search(focus, strategy, results)` → nombre de passages **nouveaux**

1. ajoutez la stratégie à `focus.tried` ;
2. ajoutez **à la fin** de `focus.passages` les passages dont le `chunk_id` est encore inconnu ;
   ne retriez pas : les scores de retrievers différents ne sont pas comparables ;
3. recalculez `focus.assessment` avec `assess_passages` (TODO 2) ;
4. si les preuves deviennent suffisantes et que `focus.winning` est vide, `focus.winning = strategy`.

```bash
python -m pytest -v -k "todo_4_"
```

## 9. TODO 5 — Contrôler une action proposée ★★

**Fonction :** `check_action(action, state, retrievers)` : lève `ActionRefused("motif")` ou ne retourne rien.

| # | Règle (dans l'ordre) | Refus si |
|---|---|---|
| 1 | focus | `action.focus` ne désigne aucun focus existant |
| 2 | `search` | retriever non autorisé ; `k > MAX_K` ; `state.retriever_calls >= MAX_RETRIEVALS` ; stratégie déjà essayée **pour ce focus** |
| 3 | `decompose` | la question a déjà été décomposée (`state.decomposed`) |
| 4 | `answer` | aucun focus n'a des preuves suffisantes |
| 5 | `abstain` | jamais |

Lisez `AgentAction` : pourquoi `k = 50` ou `retriever = "google"` passent-ils le **schéma**
Pydantic mais pas `check_action` ?

```bash
python -m pytest -v -k "todo_5_"
```

## 10. TODO 6 — Écrire la politique codée ★★★

**Fonction :** `decide_next(state)` → `AgentAction`

- **stratégies restantes** d'un focus : sa suggestion mémoire `focus.hint` d'abord (si elle n'a pas
  été essayée), puis `STRATEGIES` dans l'ordre, sans celles de `focus.tried` ;
- **focus ouvert** : preuves non suffisantes **et** au moins une stratégie restante.

| Règle | Situation | Action |
|---|---|---|
| R1 | aucun focus ouvert | `answer` si un focus est suffisant, sinon `abstain` |
| R2 | premier focus ouvert, question pas encore décomposée, preuves `partielle` | `decompose` ce focus |
| R3 | sinon | `search` ce focus avec sa première stratégie restante |

Renseignez `reason` : il apparaît dans la trace.

```bash
python -m pytest -v -k "todo_6_"
```

## 11. TODO 7 — Écrire la boucle bornée ★★★

**Fonction :** `run_agent(state, decide, toolbox, max_steps)`

Au plus `max_steps` fois : **décider** (`decide(state)`), **valider** (`validate_action`, fourni),
**contrôler** (`check_action`, TODO 5), **agir** (`execute`, fourni : il retourne l'observation).

- `ActionRefused` (refus ou sortie invalide) : `state.record(action, refusal.status, str(refusal))`
  — l'action vaut `None` si la validation a échoué — puis on continue ;
- action acceptée : `observation = execute(...)`, puis `state.record(action, "ok", observation)` ;
- arrêt après `answer` ou `abstain` ; à la fin, `return finalize(state)` (fourni).

Comparez avec `run_agent` du TD03 : qu'est-ce qui a changé ? (Indice : seul `execute`.)

```bash
python -m pytest -v -k "todo_7_"
S="Utilisation des LLM pour assister le screening des SLR en génie logiciel"
python main.py --sujet "$S" --question E3
python main.py --sujet "$S" --question E4
python main.py --sujet "$S" --question E3 --decideur llm --scenario indiscipline
```

## 12. TODO 8 — Écrire dans la mémoire épisodique ★★

**Fonction :** `remember(memory, state)` → nombre de **nouveaux** souvenirs

Un souvenir par focus dont la preuve est **`verified`** et la stratégie gagnante connue :
`Episode(question=focus.question, concepts=<libellés triés>, strategy=focus.winning)`. Si un
souvenir identique existe (mêmes concepts, même stratégie), incrémentez `uses`. Une preuve
`human_review` ou `rejected`, ou une abstention, n'écrit **rien** : c'est ce qui empêche une
mémoire empoisonnée.

```bash
python -m pytest -v -k "todo_8_"
```

## 13. TODO 9 — Relire la mémoire ★★

**Fonction :** `recall(memory, concepts, threshold)` → `Strategy` ou `None`

Similarité de **Jaccard** entre les libellés du souvenir A et ceux demandés B :
`|A ∩ B| / |A ∪ B|`. Meilleur souvenir : similarité maximale, puis `uses` maximal, puis le plus
ancien. Retournez sa stratégie si la similarité atteint `threshold` (0,5) ; sinon `None`
(mémoire vide ou absente : `None`).

```bash
python -m pytest -v -k "todo_9_"
python main.py --sujet "$S" --question E5 --oublier
python main.py --sujet "$S" --question E7
```

## 14. TODO 10 — Évaluer une exécution ★★

**Fonction :** `evaluate_run(state, annotation)` → `dict`

- *preuves* : celles de tous les focus ; *vérifiées* : statut `verified` ; *pages pertinentes* :
  pages **distinctes** `(paper_id, page)` des preuves vérifiées qui sont dans `acceptable_pages` ;
- `correct` : si `expected == "abstention"`, aucune preuve vérifiée ; sinon au moins
  `annotation.aspects` pages pertinentes ;
- `fidelite` = vérifiées / preuves ; `pertinence` = vérifiées sur une page acceptable / vérifiées
  (`None` quand le dénominateur est nul) ;
- `appels_llm`, `appels_retriever`, `pas` : compteurs de l'état ; `issue` : `state.outcome`.

```bash
python -m pytest -v -k "todo_10_"
python -m pytest -v                # tout doit être vert : 147 tests
python main.py --sujet "$S" --question E3 --evaluation
```

## 15. Observer les résultats

```bash
S="Utilisation des LLM pour assister le screening des SLR en génie logiciel"
python main.py --sujet "$S" --question E4                 # hors corpus
python main.py --sujet "$S" --question E3 --evaluation    # tableau comparatif
python main.py --sujet "$S" --question E2 --comportement invente --oublier
python main.py --sujet "$S" --rq RQ3 --oublier            # concepts du TD02
python main.py --sujet "$S" --question E3 --evaluation --decideur llm
python -m streamlit run streamlit_app.py
```

L'interface affiche, pour une question, les configurations côte à côte (issue, pas, appels,
preuves), la trace de chaque agent (action, raison, statut, observation), la mémoire épisodique
et le tableau d'évaluation. Elle écrit les mêmes artefacts que `main.py`.

Répondez :

1. E4 : le RAG naïf produit une citation `verified`. Pourquoi est-ce pourtant une mauvaise réponse ?
   Que fait l'agent, et combien cela coûte-t-il ?
2. E3 : pourquoi le naïf **et** l'avancé ne peuvent-ils pas répondre complètement, même avec un
   meilleur retriever ? Quelle observation déclenche la décomposition ?
3. Avec le décideur LLM indiscipliné, quelles propositions sont refusées et par quelle règle ? La
   réponse finale est-elle différente ? Combien d'appels LLM supplémentaires ?
4. E7 après E5 : quelle stratégie la mémoire suggère-t-elle et combien d'appels de retriever
   économise-t-elle ? Que se passe-t-il avec `--comportement invente` ?
5. RQ3 : l'agent s'abstient alors que le naïf cite une page de référence. Quel concept manque ?
   Comment l'humain peut-il corriger cela au TD02 ? Que dit ce cas sur la calibration du juge ?
6. Tableau comparatif : quelle configuration choisiriez-vous pour 10 000 questions, et à quelles
   conditions ? Pourquoi faut-il toujours rapporter le coût à côté de la qualité ?

## 16. Résultats attendus (mode démonstration, sujet de référence)

| Question (type) | Naïf | Avancé | Agentique | Agentique + mémoire |
|---|---|---|---|---|
| E1 (facile) | ✓ | ✓ | ✓ 1 recherche | ✓ 1 recherche |
| E2 (vocabulaire) | ✗ abstention | ✓ | ✓ 2 recherches | ✓ 2 recherches |
| E3 (deux aspects) | ✗ un aspect | ✗ un aspect | ✓ 3 recherches, décomposée | ✓ 1 recherche, décomposée |
| E4 (hors corpus) | ✗ répond | ✓ abstention | ✓ abstention, 4 recherches | ✓ abstention |
| E5 (vocabulaire) | ✗ abstention | ✓ | ✓ 2 recherches | ✓ 2 recherches |
| E6 (vocabulaire) | ✗ abstention | ✗ mauvaise page | ✓ 2 recherches | ✓ 2 recherches |
| E7 (reformule E5) | ✗ mauvaise page | ✗ mauvaise page | ✓ 2 recherches | ✓ 1 recherche |
| E8 (reformule E2) | ✗ abstention | ✓ | ✓ 2 recherches | ✓ 1 recherche |

| Configuration | Correctes | Fidélité | Pertinence | Appels LLM | Appels retriever | Pas |
|---|---|---|---|---|---|---|
| Naïf (TD05) | 1/8 | 1,0 | 0,5 | 0,5 | 1,0 | 2,0 |
| Avancé | 5/8 | 1,0 | 0,71 | 0,88 | 1,0 | 2,0 |
| Agentique | 8/8 | 1,0 | 1,0 | 1,12 | 2,25 | 3,38 |
| Agentique + mémoire | 8/8 | 1,0 | 1,0 | 1,12 | 1,75 | 2,88 |
| Agentique, décideur LLM (`--decideur llm`) | 8/8 | 1,0 | 1,0 | 4,5 | 2,25 | 3,38 |

Mémoire après l'évaluation : 4 souvenirs (`few_shot, rappel` → `semantique/question/k=3` ;
`cout, vote_majoritaire` → `lexical/reformulee/k=3`, utilisé 2 fois ; `decision_finale` →
`lexical/reformulee/k=3`, 2 fois ; `reproductibilite` → `lexical/reformulee/k=3`). Avec
`--comportement invente`, la preuve est `rejected`, l'issue `sans_preuve` et rien n'est mémorisé.
Tests : **147 verts**. Avec un LLM ou des embeddings réels, traces et chiffres peuvent différer :
le protocole, lui, reste le même.

## 17. Erreurs fréquentes

| Symptôme | Cause probable | Correction |
|---|---|---|
| « disagreement » couvre « agreement » | recherche de sous-chaîne (`in`) | TODO 1 : `contains_term` |
| Preuves jugées suffisantes alors que deux passages se partagent les concepts | union des passages | TODO 2 : un **seul** meilleur passage |
| `cost Cost dollars` dans la requête | doublons comparés sans casse commune | TODO 3 : `casefold()` |
| Passages dans le désordre, preuve citée sur un passage BM25 mal classé | retri par score après fusion | TODO 4 : ajouter à la fin, ne pas retrier |
| La stratégie gagnante change à chaque recherche | `winning` réécrit | TODO 4 : seulement si vide |
| L'agent répète `semantique/question/k=3` jusqu'au budget | stratégies essayées non exclues | TODO 5 et 6 |
| Une stratégie refusée sur la sous-question 1 parce qu'essayée sur la question entière | `tried` cherché dans l'état global | TODO 5 : `state.focuses[action.focus].tried` |
| L'agent décompose en boucle | `state.decomposed` ignoré | TODO 5 (règle 3) et TODO 6 (R2) |
| L'agent s'abstient alors qu'une sous-question est prouvée | R1 inversée | TODO 6 : `answer` si un focus est suffisant |
| `UnboundLocalError: action`, ou une sortie invalide tracée avec l'action du tour précédent | `action` non réinitialisée avant le `try` | TODO 7 : `action = None` au début de chaque tour |
| `issue : en_cours` | `finalize` oublié | TODO 7 |
| Souvenir écrit après une preuve `rejected` | statut non contrôlé | TODO 8 |
| La mémoire ne suggère jamais rien | Jaccard calculé sur des listes, ou seuil strict `>` | TODO 9 : ensembles, `>=` |
| E3 compté correct avec deux preuves sur la même page | pages non dédupliquées | TODO 10 : ensemble de pages |
| `ZeroDivisionError` | aucune preuve | TODO 10 : `None` |
| `ModuleNotFoundError: slrresearch` | commande lancée hors du dossier du projet | lancer depuis la racine du TD |
| Mode réel : `Connection refused` | Ollama non démarré ou modèle absent | `ollama serve`, `ollama pull nomic-embed-text` |

## 18. Prolongements facultatifs, quiz

Hors séance, sans obligation : mode réel (`cp .env.example .env`, puis
`python main.py --llm reel --embeddings ollama --decideur llm`) et comparaison des traces ;
ajout d'une stratégie (`bm25/reformulee/k=3`) à l'échelle et nouvelle évaluation ; ajout d'une
question hors corpus au jeu d'évaluation ; seuil de mémoire à 1,0 puis à 0,3. Pour relier toute
la chaîne, consultez `CONSTRUIRE_PROJET_BOUT_EN_BOUT.md`.

Les dix dernières minutes de la séance sont consacrées à un quiz individuel en ligne, qui porte
sur les notions du cours et sur le code que vous avez écrit dans les TODO 1 à 10.
