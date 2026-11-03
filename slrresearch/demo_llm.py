"""LLM de démonstration (code fourni) : reproductible, sans clé ni réseau.

Il imite un ChatModel LangChain (méthode invoke -> objet avec .content) et
répond selon le type de prompt reçu :
- prompt few-shot du TD02          -> trois RQ conformes ;
- prompt de concepts du TD02       -> un JSON de concepts ;
- prompt de décision du niveau 3   -> une action, selon un scénario ReAct ;
- prompt de screening du TD04      -> le jugement prévu pour l'article
  (data/jeu_etiquete_td04.json et data/resumes_td04.json) ;
- prompt de décision RAG du TD06  -> une action de l'agent RAG (même scénarios) ;
- sortie structurée SubQuestions (TD06) -> découpe une question à plusieurs aspects
  (« ... et combien ... ? ») en sous-questions ;
- sortie structurée du TD05 (with_structured_output) -> une preuve proposée
  à partir du CONTEXTE DOCUMENTAIRE, selon un comportement RAG :
    "fidele"       recopie exactement une phrase du meilleur passage ;
    "approximatif" la recopie en ajoutant un mot (citation presque exacte) ;
    "invente"      produit une citation absente du passage.

Scénarios ReAct (TD03) et décideur RAG (TD06) :
- "discipline"   : propose des actions valides et termine proprement ;
- "indiscipline" : se trompe d'outil, répond hors contrat, oublie un champ,
                   répète une action... puis finit par chercher et terminer.
                   Au TD06 : répond trop tôt, invente un retriever, écrit du
                   texte libre, demande k = 50, puis suit l'échelle des stratégies.
"""
import json
import re
from pathlib import Path
from types import SimpleNamespace

DATA = Path(__file__).resolve().parent.parent / "data"

DEMO_QUESTIONS = (
    "RQ1. Comment les LLM améliorent-ils le rappel du screening des SLR ?\n"
    "RQ2. Quel contrôle humain doit-il être conservé lors du screening assisté ?\n"
    "RQ3. Quelles métriques évaluent-elles la qualité du screening automatisé ?"
)

LLM_TERMS = {"label": "large language models",
             "terms": ["large language models", "LLM", "generative AI"]}
SCREENING_TERMS = {"label": "screening",
                   "terms": ["screening", "study selection", "title and abstract screening"]}
DEMO_CONCEPTS = {
    "rappel": [LLM_TERMS, SCREENING_TERMS, {"label": "recall", "terms": ["recall", "sensitivity"]}],
    "humain": [{"label": "human oversight", "terms": ["human oversight", "human-in-the-loop"]},
               SCREENING_TERMS, LLM_TERMS],
    "métriques": [{"label": "evaluation metrics",
                   "terms": ["evaluation metrics", "precision", "work saved over sampling"]},
                  SCREENING_TERMS, {"label": "automation", "terms": ["automation", "automated"]}],
}

REACT_SCENARIOS = ("discipline", "indiscipline")
RAG_BEHAVIOURS = ("fidele", "approximatif", "invente")
INVENTED_QUOTE = "LLM screening always reaches perfect recall without any human review."
HEADER = re.compile(r"^\[PAPER (?P<paper>.+?)\] \[CHUNK (?P<chunk>.+?)\] \[PAGE (?P<page>\d+)\]$", re.MULTILINE)


def _text_of(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(str(getattr(message, "content", message)) for message in prompt)


def _state_of(text: str) -> dict:
    match = re.search(r"\[ÉTAT\]\n(\{.*?\})\n\[", text, re.DOTALL)
    return json.loads(match.group(1)) if match else {}


class DemoChatModel:
    """Faux ChatModel déterministe pour la démonstration et les tests."""

    def __init__(self, react_scenario: str = "discipline", rag_behaviour: str = "fidele"):
        if react_scenario not in REACT_SCENARIOS:
            raise ValueError(f"Scénario ReAct inconnu : {react_scenario}")
        if rag_behaviour not in RAG_BEHAVIOURS:
            raise ValueError(f"Comportement RAG inconnu : {rag_behaviour}")
        self.react_scenario = react_scenario
        self.rag_behaviour = rag_behaviour
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        """Imite l'API LangChain : un objet dont invoke() retourne une instance de schema."""
        return _StructuredDemo(self, schema)

    def invoke(self, prompt):
        text = _text_of(prompt)
        self.prompts.append(text)
        return SimpleNamespace(content=self._answer(text))

    # -- réponses ----------------------------------------------------------
    def _answer(self, text: str) -> str:
        if "[SCREENING]" in text:
            return self._screen(text)
        if "[DÉCISION RAG]" in text:
            return self._decide_rag(_state_of(text))
        if "[DÉCISION]" in text:
            return self._decide(_state_of(text))
        if "Tu prépares une recherche bibliographique" in text:
            question = text.split("[QUESTION]\n", 1)[-1].split("\n", 1)[0]
            for key, concepts in DEMO_CONCEPTS.items():
                if key in question:
                    return json.dumps({"concepts": concepts}, ensure_ascii=False)
            return json.dumps({"concepts": [LLM_TERMS, SCREENING_TERMS]}, ensure_ascii=False)
        return DEMO_QUESTIONS

    def _screen(self, text: str) -> str:
        match = re.search(r"^id : (.+)$", text, re.MULTILINE)
        key = match.group(1).strip() if match else ""
        scripted = {}
        gold = DATA / "jeu_etiquete_td04.json"
        if gold.exists():
            scripted.update({p["id"]: p["llm_demo"] for p in json.loads(gold.read_text(encoding="utf-8"))["papers"]})
        abstracts = DATA / "resumes_td04.json"
        if abstracts.exists():
            scripted.update({k: v["llm_demo"] for k, v in
                             json.loads(abstracts.read_text(encoding="utf-8"))["abstracts"].items()})
        answer = scripted.get(key)
        if answer is None:
            answer = {"decision": "uncertain", "confidence": 0.3,
                      "reasons": ["article inconnu du LLM de démonstration"], "evidence": []}
        if isinstance(answer, str):
            return answer
        return "```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```"

    def _decide(self, state: dict) -> str:
        queries = state.get("requetes_td02", {})
        step = state.get("pas_utilises", 0)
        if self.react_scenario == "indiscipline":
            script = [
                {"action": "search", "tool": "scholar", "query": queries.get("scholar", "")},
                "Je pense qu'il faudrait d'abord chercher sur OpenAlex.",
                {"action": "finish", "tool": "openalex"},
                {"action": "search", "tool": "openalex", "query": queries.get("scholar", ""), "limit": 10},
                {"action": "search", "tool": "openalex", "query": queries.get("scholar", ""), "limit": 10},
                {"action": "search", "tool": "crossref", "query": queries.get("ieee", ""), "limit": 10},
                {"action": "finish"},
            ]
            choice = script[min(step, len(script) - 1)]
            return choice if isinstance(choice, str) else json.dumps(choice, ensure_ascii=False)
        # Scénario discipliné : suit le plan, relance une fois, puis termine.
        history = state.get("historique", [])
        called = [item["outil"] for item in history]
        last = history[-1] if history else None
        if last and last["statut"] == "rate_limited" and called.count(last["outil"]) == 1:
            base = {"openalex": "scholar", "crossref": "ieee",
                    "semantic_scholar": "acm"}.get(last["outil"], "scholar")
            action = {"action": "search", "tool": last["outil"], "query": queries.get(base, "")}
        else:
            for tool, base in (("openalex", "scholar"), ("crossref", "ieee"),
                               ("semantic_scholar", "acm")):
                if tool not in called and base in queries:
                    action = {"action": "search", "tool": tool, "query": queries[base], "limit": 10}
                    break
            else:
                if state.get("articles_uniques", 0) < 3 and "openalex_broad" not in called:
                    action = {"action": "search", "tool": "openalex_broad",
                              "query": queries.get("scholar", "")}
                else:
                    action = {"action": "finish"}
        return "```json\n" + json.dumps(action, ensure_ascii=False) + "\n```"


    def _decide_rag(self, view: dict) -> str:
        """TD06 : action d'un agent RAG, à partir de la vue de l'état."""
        if self.react_scenario == "indiscipline":
            script = [
                {"action": "answer"},
                {"action": "search", "focus": 0, "retriever": "google", "query_kind": "question", "k": 3},
                "Je vais d'abord relire les passages les plus récents.",
                {"action": "search", "focus": 0, "retriever": "semantique", "query_kind": "question", "k": 50},
            ]
            step = view.get("pas_utilises", 0)
            if step < len(script):
                choice = script[step]
                return choice if isinstance(choice, str) else json.dumps(choice)
        ladder = view.get("echelle_strategies", [])

        def remaining(focus: dict) -> list[dict]:
            ordered = ([focus["suggestion_memoire"]] if focus.get("suggestion_memoire") else []) + ladder
            result = []
            for strategy in ordered:
                if strategy not in focus.get("essayees", []) and strategy not in result:
                    result.append(strategy)
            return result

        focuses = view.get("focus", [])
        open_focus = [f for f in focuses if f.get("preuves") != "suffisante" and remaining(f)]
        if not open_focus:
            done = any(f.get("preuves") == "suffisante" for f in focuses)
            action = {"action": "answer" if done else "abstain"}
        elif not view.get("decomposee") and open_focus[0].get("preuves") == "partielle":
            action = {"action": "decompose", "focus": open_focus[0]["index"],
                      "reason": "preuves partielles"}
        else:
            focus = open_focus[0]
            action = {"action": "search", "focus": focus["index"], **remaining(focus)[0],
                      "reason": "stratégie suivante"}
        return "```json\n" + json.dumps(action, ensure_ascii=False) + "\n```"


INTERROGATIVES = r"(?:combien|quel|quelle|quels|quelles|comment|pourquoi|qui|à qui|où|que)\b"


def _split_question(question: str) -> list[str]:
    """TD06 : découpe déterministe « A et <mot interrogatif> B ? » -> [« A ? », « B ? »]."""
    parts = re.split(r"\s+et\s+(?=" + INTERROGATIVES + ")", question.strip(), flags=re.IGNORECASE)
    if len(parts) < 2:
        return [question.strip()]
    clean = []
    for part in parts:
        part = part.strip().rstrip(" ?").strip()
        clean.append(part[:1].upper() + part[1:] + " ?")
    return clean


# ---------------------------------------------------------------------------
# TD05 : sortie structurée simulée (preuve proposée à partir du contexte)
# ---------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]


class _StructuredDemo:
    def __init__(self, model: DemoChatModel, schema):
        self.model, self.schema = model, schema

    def invoke(self, prompt):
        from slrresearch.demo_embeddings import DemoEmbeddings

        text = _text_of(prompt)
        self.model.prompts.append(text)
        question = text.split("[QUESTION]\n", 1)[-1].split("\n", 1)[0]
        if getattr(self.schema, "__name__", "") == "SubQuestions":
            return self.schema(sub_questions=_split_question(question))
        context = text.split("<<<", 1)[-1].rsplit(">>>", 1)[0]
        headers = list(HEADER.finditer(context))
        if not headers:
            raise ValueError("Contexte documentaire sans en-tête [PAPER] [CHUNK] [PAGE]")
        embeddings = DemoEmbeddings()
        target = embeddings.embed_query(question)

        # TD06 : la question peut être suivie des termes du corpus « (termes du corpus : ...) ».
        hint = re.search(r"\(termes du corpus : (.+)\)\s*$", question)
        hint_words = set(re.findall(r"[a-z0-9@-]+", hint.group(1).casefold())) if hint else set()

        def score(sentence: str) -> tuple:
            """(TD06 : termes du corpus présents), concepts couverts, résultat chiffré, cosinus."""
            vector = embeddings.embed_query(sentence)
            covered = sum(1 for a, b in zip(target, vector) if a and b)
            dot = sum(a * b for a, b in zip(target, vector))
            norm = (sum(a * a for a in target) * sum(b * b for b in vector)) ** 0.5
            hits = len(hint_words & set(re.findall(r"[a-z0-9@-]+", sentence.casefold())))
            return hits, covered, bool(re.search(r"\d", sentence)), dot / norm if norm else 0.0

        # Comme un LLM « appliqué », il choisit dans TOUS les passages transmis la phrase
        # la plus proche de la question ; à égalité, le premier passage l'emporte.
        options = []
        for index, header in enumerate(headers):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(context)
            passage = context[header.end():end].strip()
            for sentence in _sentences(passage) or [passage]:
                options.append((score(sentence), -index, header, sentence))
        _, _, first, sentence = max(options, key=lambda option: (option[0], option[1]))
        words = sentence.split()
        if self.model.rag_behaviour == "approximatif":
            if len(words) >= 10:
                quote = " ".join(words[:2] + ["reportedly"] + words[2:])
            else:
                last = words[-1].rstrip(".,;:")
                quote = " ".join(words[:-1] + [last + "s" + words[-1][len(last):]])
        elif self.model.rag_behaviour == "invente":
            quote = INVENTED_QUOTE
        else:
            quote = sentence
        return self.schema(
            answer=f"Selon le passage {first['chunk']} (page {first['page']}) : {sentence}",
            supporting_quote=quote,
            paper_id=first["paper"],
            chunk_id=first["chunk"],
            page=int(first["page"]),
        )
