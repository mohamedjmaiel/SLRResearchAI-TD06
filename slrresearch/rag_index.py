"""Indexation et classement (code fourni, à lire).

Phase 1 OFFLINE - avant toute question, on indexe le corpus UNE fois :
    build_lexical_index(chunks)             normaliser + tokeniser, index inversé, vecteurs TF-IDF
    build_vector_index(chunks, embeddings)  un embedding par chunk, conservé en mémoire

Phase 2 ONLINE - à chaque question, on CONSULTE l'index (TODO 3 et 5 dans rag_pipeline.py).

Outils de classement fournis :
    top_k(scored, k)            les k meilleurs passages candidats
    bm25_retrieve(index, q, k)  BM25 : fréquence + rareté, avec saturation et longueur
    rrf_fuse(classements, k)    fusion hybride Reciprocal Rank Fusion (aperçu du TD06)
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from slrresearch.rag_models import Chunk

BM25_K1 = 1.5        # saturation : au-delà de quelques occurrences, un terme répété rapporte peu
BM25_B = 0.75        # normalisation par la longueur : un long passage n'est pas favorisé
RRF_CONSTANT = 60    # constante usuelle de la Reciprocal Rank Fusion


def tokenize(text: str) -> list[str]:
    """Normalisation (minuscules) + tokenisation : mots d'au moins 2 caractères.

    Ni suppression des mots vides ni lemmatisation : l'IDF donne déjà un poids
    faible aux mots présents partout (« the », « and ») ; « review » et
    « reviews » restent deux tokens différents (limite assumée du TD).
    """
    return re.findall(r"[a-zA-ZÀ-ÿ0-9@]{2,}", text.casefold())


def sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    """Cosinus entre deux vecteurs creux {terme: poids}."""
    dot = sum(left[word] * right[word] for word in set(left) & set(right))
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


# ---------------------------------------------------------------------------
# Index lexical
# ---------------------------------------------------------------------------

@dataclass
class LexicalIndex:
    """Index construit une fois ; consulté à chaque question."""

    chunks: list[Chunk]
    postings: dict[str, set[int]]            # index inversé : token -> positions des chunks
    term_counts: list[Counter]               # fréquences des tokens dans chaque chunk (TF)
    document_frequency: dict[str, int]       # nombre de chunks contenant chaque token
    vectors: list[dict[str, float]] = field(default_factory=list)   # vecteurs TF-IDF des chunks

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def average_length(self) -> float:
        return sum(sum(c.values()) for c in self.term_counts) / self.size if self.size else 0.0

    def idf(self, term: str) -> float:
        """Rareté lissée : un terme présent partout pèse peu, un terme rare pèse beaucoup."""
        return math.log((1 + self.size) / (1 + self.document_frequency.get(term, 0))) + 1

    def tfidf(self, counts: Counter) -> dict[str, float]:
        return {term: count * self.idf(term) for term, count in counts.items()}

    def query_vector(self, query: str) -> dict[str, float]:
        """Vecteur TF-IDF de la requête, calculé avec les statistiques du CORPUS."""
        return self.tfidf(Counter(tokenize(query)))


def build_lexical_index(chunks: list[Chunk]) -> LexicalIndex:
    """OFFLINE : tokenise chaque chunk, construit l'index inversé et les vecteurs TF-IDF."""
    term_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
    postings: dict[str, set[int]] = {}
    for position, counts in enumerate(term_counts):
        for term in counts:
            postings.setdefault(term, set()).add(position)
    index = LexicalIndex(chunks=list(chunks), postings=postings, term_counts=term_counts,
                         document_frequency={term: len(ids) for term, ids in postings.items()})
    index.vectors = [index.tfidf(counts) for counts in term_counts]
    return index


# ---------------------------------------------------------------------------
# Index vectoriel
# ---------------------------------------------------------------------------

@dataclass
class VectorIndex:
    """Embeddings des chunks, calculés une fois avec le modèle de l'index."""

    chunks: list[Chunk]
    vectors: list[list[float]]
    model: str


def build_vector_index(chunks: list[Chunk], embeddings) -> VectorIndex:
    """OFFLINE : un seul appel embed_documents sur tout le corpus."""
    vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
    return VectorIndex(chunks=list(chunks), vectors=vectors, model=type(embeddings).__name__)


# ---------------------------------------------------------------------------
# Classement
# ---------------------------------------------------------------------------

def top_k(scored: list[dict], k: int) -> list[dict]:
    """Les k meilleurs passages CANDIDATS (score > 0), score décroissant, égalités par chunk_id."""
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1")
    candidates = [item for item in scored if item["retrieval_score"] > 0]
    return sorted(candidates, key=lambda item: (-item["retrieval_score"], item["chunk_id"]))[:k]


def bm25_retrieve(index: LexicalIndex, query: str, k: int, k1: float = BM25_K1,
                  b: float = BM25_B) -> list[dict]:
    """BM25 sur le MÊME index que TF-IDF.

    score(q, d) = Σ idf(t) · tf(t, d) · (k1 + 1) / (tf(t, d) + k1 · (1 − b + b · |d| / |d|moyen))
    - saturation : quand tf grandit, la fraction tend vers (k1 + 1), elle ne croît plus ;
    - longueur   : |d| > |d|moyen augmente le dénominateur, donc baisse le score.
    """
    terms = set(tokenize(query))
    candidates = set().union(*(index.postings.get(term, set()) for term in terms)) if terms else set()
    n, average = index.size, index.average_length or 1.0
    scored = []
    for position in candidates:
        counts = index.term_counts[position]
        length = sum(counts.values())
        score = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = index.document_frequency[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        scored.append({**index.chunks[position].model_dump(), "retrieval_score": score})
    return top_k(scored, k)


def rrf_fuse(rankings: list[list[dict]], k: int, constant: int = RRF_CONSTANT) -> list[dict]:
    """Hybride : score RRF = Σ 1 / (constant + rang) sur les classements fournis."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            entry = fused.setdefault(item["chunk_id"], {**item, "retrieval_score": 0.0})
            entry["retrieval_score"] += 1 / (constant + rank)
    return top_k(list(fused.values()), k)
