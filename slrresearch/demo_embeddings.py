"""Modèle d'embeddings de démonstration (code fourni, à lire).

Un vrai modèle d'embeddings (ex. nomic-embed-text via Ollama) APPREND à placer
les textes de sens proche dans des régions voisines d'un espace vectoriel.
Pour travailler sans réseau et de façon reproductible, ce modèle pédagogique
fixe cet espace à la main : chaque dimension est un CONCEPT, et un lexique
bilingue (anglais / français) associe des mots à ces concepts.

    "recall", "sensitivity", "rappel"  ->  même dimension « recall »

Conséquence observable : il rapproche des passages qui n'ont AUCUN mot en
commun avec la requête (synonymes, autre langue), ce que TF-IDF ne sait pas
faire. Il a aussi les défauts d'un lexique figé : un mot absent du lexique est
invisible. Il expose la même interface que les embeddings LangChain :
embed_query(texte) et embed_documents(textes).
"""
import re

DIMENSIONS: dict[str, set[str]] = {
    "llm": {"llm", "llms", "language", "models", "model", "generative", "gpt", "chatgpt",
            "modèle", "modèles", "langage", "génératif", "générative"},
    "screening": {"screening", "screen", "screener", "screeners", "selection", "select",
                  "eligibility", "eligible", "abstracts", "titles", "sélection", "tri",
                  "éligibilité", "criteria", "critères"},
    "recall": {"recall", "sensitivity", "missed", "rappel", "sensibilité", "retrieves",
               "identify", "manqués"},
    "metrics": {"precision", "specificity", "wss", "metrics", "metric", "measure", "measured",
                "evaluation", "évaluation", "métriques", "mesure", "mesurer", "précision",
                "proportion", "qualité", "quality", "report"},
    "human": {"human", "humans", "humain", "humaine", "reviewer", "reviewers", "relecteur",
              "relecteurs", "oversight", "supervision", "control", "contrôle", "audited",
              "audit", "loop", "person", "people", "team", "escalated", "conservé"},
    "prompting": {"prompt", "prompts", "prompting", "shot", "chain", "thought", "voting",
                  "reasoning", "stratégie", "stratégies", "strategies", "strategy"},
    "effort": {"workload", "effort", "cost", "costs", "coût", "dollars", "time", "reading",
               "charge", "saves", "saved", "reduced"},
    "risk": {"wrong", "error", "errors", "hallucination", "hallucinated", "bias", "limitations",
             "threatens", "leakage", "reproducibility", "erreur", "risque", "limites"},
    "review": {"systematic", "review", "reviews", "slr", "slrs", "revue", "revues",
               "literature", "littérature", "studies", "study", "primary"},
}
TOKEN = re.compile(r"[a-zA-ZÀ-ÿ0-9@]+")


class DemoEmbeddings:
    """Embeddings déterministes : un vecteur de comptes par concept (non normalisé)."""

    dimensions = list(DIMENSIONS)

    def _embed(self, text: str) -> list[float]:
        words = TOKEN.findall(text.casefold())
        return [float(sum(word in lexicon for word in words)) for lexicon in DIMENSIONS.values()]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]
