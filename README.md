# SLRResearchAI — TD06 RAG agentique

Ce TD transforme le pipeline RAG classique du TD05 en **agent** : il observe si les preuves
retrouvées suffisent, choisit de reformuler, de changer de retriever, d'élargir le Top-k, de
décomposer la question, de répondre ou de s'abstenir, dans une boucle bornée et contrôlée. Une
mémoire épisodique conserve les stratégies qui ont produit des preuves vérifiées. Les
configurations naïve, avancée et agentique sont comparées sur le même corpus et les mêmes
questions annotées, qualité et coût compris.

Le dépôt conserve les modules acquis pendant les TD01 à TD05 dans le même package
`slrresearch`. Consultez `CONSTRUIRE_PROJET_BOUT_EN_BOUT.md` pour relier les exécutions.

L'énoncé complet est dans `ENONCE_TD06.md`. Le seul fichier à modifier est
`slrresearch/agentic_rag.py` (TODO 1 à 10).

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -v                    # 112 verts / 35 rouges au départ, 147 verts à la fin
python main.py                         # saisissez le sujet de la revue
python -m streamlit run streamlit_app.py
```

Le mode démonstration fonctionne sans clé ni réseau. Pour le mode réel :
`cp .env.example .env`, compléter le fichier, puis
`python main.py --llm reel --embeddings ollama --decideur llm`.
