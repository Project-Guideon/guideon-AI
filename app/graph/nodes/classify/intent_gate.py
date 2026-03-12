from __future__ import annotations

import json

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

_ALLOWED_INTENTS = {"rag", "smalltalk", "event", "struct_db"}
_DEFAULT_RANKING = ["rag", "smalltalk", "event", "struct_db"]


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
                    "Classify the user input into ALL of these 4 categories, ranked by likelihood:\n"
                    "  rag       : questions about history, origin, explanation, stories, descriptions, cultural info\n"
                    "  smalltalk : greetings, emotions, casual chat, self-introduction, jokes\n"
                    "  event     : questions about current/upcoming events, festivals, performances, exhibitions\n"
                    "  struct_db : operational info (hours, price, tickets, reservation, contact, rules, restroom, parking, facilities)\n"
                    "\n"
                    "Respond ONLY with valid JSON. Example:\n"
                    '{"ranking": ["rag", "struct_db", "event", "smalltalk"]}\n'
                    "\n"
                    "The first item is the most likely intent. Include all 4 categories in the ranking.\n"
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
