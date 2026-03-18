from __future__ import annotations

import json

from openai import APIConnectionError, APITimeoutError, RateLimitError, APIError

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState

_ALLOWED_INTENTS = {"rag", "smalltalk", "event", "struct_db"}
_DEFAULT_RANKING = ["rag","struct_db", "smalltalk", "event"]
_MAX_INTENTS = 2  # 상위 N개 의도만 시도 (1~4)
_ALLOWED_PLACE_CATEGORIES = {
    "TOILET", "TICKET", "RESTAURANT", "SHOP", "INFO", "ATTRACTION", "PARKING", "OTHER"
}


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
                    '{"ranking": ["struct_db", "rag", "event", "smalltalk"], "place_category": "TOILET"}\n'
                    "\n"
                    "The first item is the most likely intent. Include all 4 categories.\n"
                    "place_category: set ONLY when top intent is struct_db, otherwise null.\n"
                    "  Possible values: TOILET, TICKET, RESTAURANT, SHOP, INFO, ATTRACTION, PARKING, OTHER\n"
                    "DO NOT generate any answer or explanation."
                ),
            },
            {"role": "user", "content": f"Classify this input: {text}"},
        ]

        method = "llm"
        error = ""
        try:
            raw = llm.chat(messages, max_tokens=60)
            parsed = json.loads(raw)
            ranking = parsed.get("ranking", _DEFAULT_RANKING)
            # 1순위가 struct_db일 때만 place_category 적용, 허용 enum 외 값은 None
            raw_category = parsed.get("place_category")
            place_category = (
                raw_category.upper()
                if isinstance(raw_category, str) and raw_category.upper() in _ALLOWED_PLACE_CATEGORIES
                else None
            )

            # 유효성 검증: 4개 의도가 모두 포함되어야 함
            if not isinstance(ranking, list):
                ranking = _DEFAULT_RANKING
                method = "llm_fallback_default"
            else:
                # 정규화(strip/lower) → 중복 제거(순서 유지) → 허용 필터링
                seen: set[str] = set()
                normalized: list[str] = []
                for r in ranking:
                    s = r.strip().lower() if isinstance(r, str) else ""
                    if s and s not in seen:
                        seen.add(s)
                        normalized.append(s)
                valid = [r for r in normalized if r in _ALLOWED_INTENTS]
                missing = [i for i in _DEFAULT_RANKING if i not in valid]
                ranking = valid + missing
                if len(valid) < len(_ALLOWED_INTENTS):
                    method = "llm_partial_fix"
        except json.JSONDecodeError as e:
            ranking = _DEFAULT_RANKING
            place_category = None
            method = "llm_fallback_invalid_json"
            error = str(e)
        except (APIConnectionError, APITimeoutError) as e:
            ranking = _DEFAULT_RANKING
            place_category = None
            method = "llm_fallback_connection_error"
            error = str(e)
        except RateLimitError as e:
            ranking = _DEFAULT_RANKING
            place_category = None
            method = "llm_fallback_rate_limit"
            error = str(e)
        except APIError as e:
            ranking = _DEFAULT_RANKING
            place_category = None
            method = "llm_fallback_api_error"
            error = f"{e.__class__.__name__}: {e}"

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("intent_gate")
        trace["_flow"] = flow
        trace["intent_gate"] = {
            "text": text,
            "ranking": ranking,
            "place_category": place_category,
            "method": method,
        }
        if error:
            trace["intent_gate"]["error"] = error

        # 상위 N개 의도만 시도
        ranking = ranking[:_MAX_INTENTS]

        # 1순위가 struct_db가 아니면 place_category 무효화
        if ranking[0] != "struct_db":
            place_category = None

        return {
            "intent_ranking": ranking,
            "current_intent_index": 0,
            "place_category": place_category,
            "trace": trace,
        }

    return intent_gate_node
