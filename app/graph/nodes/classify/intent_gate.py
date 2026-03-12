from __future__ import annotations

import json

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

_ALLOWED_INTENTS = {"rag", "smalltalk", "event", "struct_db"}
_DEFAULT_RANKING = ["rag","struct_db", "smalltalk", "event"]


def make_intent_gate_node(llm: OpenAILLM):
    """LLM-only 의도 분류 노드 팩토리.

    4개 의도(rag, smalltalk, event, struct_db)를 순위로 반환.
    키워드 룰 없이 LLM 한 번 호출로 분류 + 순위 결정.
    """

    def intent_gate_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intent classifier for a multilingual tourism voice chatbot.\n"
                    "Classify the user input by WHAT THE USER WANTS, not by the topic or keywords.\n"
                    "Rank ALL 4 categories by likelihood:\n"
                    "\n"
                    "  rag       : The user wants to UNDERSTAND or LEARN something — history, origin,\n"
                    "              meaning, background stories, cultural significance, how/why something\n"
                    "              was built, what something used to be, architectural details, legends.\n"
                    "              (Even if the subject is a facility like a restroom or gate, if the user\n"
                    "              asks about its history or meaning, this is rag.)\n"
                    "  smalltalk : The user is making casual conversation — greetings, emotions, jokes,\n"
                    "              self-introduction, thanks, or not asking for any specific information.\n"
                    "  event     : The user wants to know about TIME-BOUND happenings — current or\n"
                    "              upcoming festivals, performances, exhibitions, seasonal programs,\n"
                    "              special openings, scheduled activities.\n"
                    "  struct_db : The user wants to FIND A SPECIFIC PLACE or LOCATION — where is\n"
                    "              a restroom, parking lot, specific building, gate, ticket booth, shop,\n"
                    "              restaurant, or any named place within the site. The user needs\n"
                    "              directions or location of a place stored in the database.\n"
                    "\n"
                    "Key distinction: 'Where is the restroom?' → struct_db (finding a place)\n"
                    "                 'What was the restroom area used for historically?' → rag (learning)\n"
                    "\n"
                    "Respond ONLY with valid JSON. Example:\n"
                    '{"ranking": ["rag", "struct_db", "event", "smalltalk"]}\n'
                    "\n"
                    "The first item is the most likely intent. Include all 4 categories.\n"
                    "DO NOT generate any answer or explanation."
                ),
            },
            {"role": "user", "content": f"Classify this input: {text}"},
        ]

        method = "llm"
        try:
            raw = llm.chat(messages, max_tokens=60)
            parsed = json.loads(raw)
            ranking = parsed.get("ranking", _DEFAULT_RANKING)

            # 유효성 검증: 4개 의도가 모두 포함되어야 함
            if not isinstance(ranking, list):
                ranking = _DEFAULT_RANKING
                method = "llm_fallback_default"
            else:
                # 허용된 의도만 필터링 + 누락된 의도 추가
                valid = [r for r in ranking if r in _ALLOWED_INTENTS]
                missing = [i for i in _DEFAULT_RANKING if i not in valid]
                ranking = valid + missing
                if len(valid) < len(_ALLOWED_INTENTS):
                    method = "llm_partial_fix"
        except Exception:
            ranking = _DEFAULT_RANKING
            method = "llm_fallback_default"

        trace = dict(state.get("trace") or {})
        trace["intent_gate"] = {
            "text": text,
            "ranking": ranking,
            "method": method,
        }

        return {
            "intent_ranking": ranking,
            "current_intent_index": 0,
            "trace": trace,
        }

    return intent_gate_node
