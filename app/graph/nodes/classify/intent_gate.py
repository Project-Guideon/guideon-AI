from __future__ import annotations

import json
from typing import Optional

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
    cross-language STT 오류를 감지해 actual_language / user_language 보정.
    """

    def intent_gate_node(state: GraphState) -> dict:
        text: str = state.get("normalized_text", "")
        raw_stt_language: str = state.get("language_code") or state.get("user_language") or "ko"
        stt_language: str = raw_stt_language.split("-")[0].lower() if isinstance(raw_stt_language, str) else "ko"

        # 최근 2턴(4개 메시지)만 — 대명사/지시어 해소용
        raw_history: list = state.get("chat_history") or []
        recent_history = raw_history[-4:] if len(raw_history) > 4 else raw_history
        history_block = ""
        if recent_history:
            lines = []
            for msg in recent_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"user: {content}")
                elif role == "assistant":
                    lines.append(f"assistant: {content}")
            if lines:
                history_block = "\n\nRecent conversation (for coreference only — do NOT answer based on this):\n" + "\n".join(lines)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant for a multilingual tourism voice chatbot.\n"
                    "Do FOUR things with the user input:\n"
                    "\n"
                    "1. DETECT actual spoken language (actual_language):\n"
                    "   Determine the language in which the input text is actually written.\n"
                    "   Return the ISO 639-1 code: ko, en, ja, or zh.\n"
                    "   Use stt_language as a reference, but OVERRIDE it when the text is clearly\n"
                    "   in a different language (e.g., stt_language='ko' but text is entirely English).\n"
                    "   Special case — cross-language phonetic confusion:\n"
                    "     When speech starts with a proper noun from language A, STT sometimes\n"
                    "     transcribes the ENTIRE sentence in language A's script even though the\n"
                    "     speaker was using language B. Detect this by looking for:\n"
                    "     - Japanese→Korean: Korean particles as stand-ins for Japanese particles\n"
                    "       (도≈の, 은/는≈は), Korean phonetic endings resembling Japanese (입니까≈ですか,\n"
                    "       입니다≈です), Sino-Korean in Japanese grammatical flow.\n"
                    "     - English→Korean: English phonemes rendered as similar-sounding Korean\n"
                    "       syllables, English word order in Korean script.\n"
                    "     In these cases set actual_language to the true spoken language (ja/en/etc.).\n"
                    "   Base judgment on the text content itself — do not use domain knowledge.\n"
                    "\n"
                    "2. CORRECT STT (speech-to-text) errors:\n"
                    "   - Fix misheard words, phonetically similar substitutions, garbled text\n"
                    "   - Preserve the language identified in step 1 (actual_language)\n"
                    "   - If actual_language differs from stt_language, rewrite corrected_text in\n"
                    "     actual_language (e.g., reconstruct the likely original Japanese utterance)\n"
                    "   - If no correction needed, return the input text unchanged\n"
                    "\n"
                    "3. CLASSIFY intent — rank ALL 4 categories by likelihood:\n"
                    "  rag       : The user wants to UNDERSTAND or LEARN something — history, origin,\n"
                    "              meaning, background stories, cultural significance, how/why something\n"
                    "              was built, what something used to be, architectural details, legends,\n"
                    "              prices, admission fees, costs, or any numerical information.\n"
                    "              (Even if the subject is a facility like a restroom or gate, if the user\n"
                    "              asks about its history or meaning, this is rag.)\n"
                    "  smalltalk : The user is making casual conversation — greetings, emotions, jokes,\n"
                    "              self-introduction, thanks, or not asking for any specific information.\n"
                    "  event     : The user wants to know about TIME-BOUND happenings — current or\n"
                    "              upcoming festivals, performances, exhibitions, seasonal programs,\n"
                    "              special openings, scheduled activities.\n"
                    "  struct_db : The user wants to FIND A SPECIFIC PLACE or LOCATION — where is\n"
                    "              a restroom, parking lot, specific building, gate, ticket booth, shop,\n"
                    "              restaurant, or any named place within the site.\n"
                    "\n"
                    "Key distinction: 'Where is the restroom?' → struct_db (finding a place)\n"
                    "                 'What was the restroom area used for historically?' → rag (learning)\n"
                    "\n"
                    "4. TRANSLATE to Korean (retrieval_query_ko):\n"
                    "   - Translate the SEMANTIC MEANING of the input into Korean keywords\n"
                    "   - Do NOT interpret, summarize, or add any words not present in the original\n"
                    "   - If the input is already Korean, return it as-is\n"
                    "\n"
                    "5. EXTRACT place name (place_name_query):\n"
                    "   - ONLY when top intent is struct_db AND the user mentions a SPECIFIC named place\n"
                    "     (e.g., '광화문', '경회루', '근정전', a named shop or building)\n"
                    "   - Set to null if the user asks for a category only (e.g., '화장실 어디야?', '주차장')\n"
                    "   - Set to null if top intent is NOT struct_db\n"
                    "   - Write in Korean (translate if needed)\n"
                    "\n"
                    "Respond ONLY with valid JSON. Example:\n"
                    '{"actual_language": "ja", "corrected_text": "경복궁の観覧時間は何時から何時までですか", '
                    '"ranking": ["rag", "struct_db", "smalltalk", "event"], "place_category": null, '
                    '"place_name_query": null, "retrieval_query_ko": "경복궁 관람 시간"}\n'
                    "\n"
                    "actual_language: ISO 639-1 code — ko, en, ja, or zh.\n"
                    "The first item in ranking is the most likely intent. Include all 4 categories.\n"
                    "place_category: set ONLY when top intent is struct_db, otherwise null.\n"
                    "  Possible values: TOILET, TICKET, RESTAURANT, SHOP, INFO, ATTRACTION, PARKING, OTHER\n"
                    "place_name_query: specific named place in Korean, or null.\n"
                    "retrieval_query_ko: ALWAYS required, Korean keywords only.\n"
                    "DO NOT generate any answer or explanation."
                ),
            },
            {"role": "user", "content": f"Input: {text}\nstt_language: {stt_language}{history_block}"},
        ]

        method = "llm"
        error = ""
        corrected_text = text      # 기본값: 보정 없음
        retrieval_query_ko = text  # 기본값: 원문 그대로
        actual_language = stt_language  # 기본값: STT 감지 언어 그대로
        place_name_query: Optional[str] = None
        try:
            raw = llm.chat(messages, max_tokens=300)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"LLM response is not a JSON object: {type(parsed).__name__}")

            # 실제 발화 언어 감지 (cross-language STT 오류 보정)
            al = parsed.get("actual_language")
            if isinstance(al, str) and al.strip().lower() in {"ko", "en", "ja", "zh"}:
                actual_language = al.strip().lower()

            # STT 보정 텍스트 추출
            ct = parsed.get("corrected_text")
            if isinstance(ct, str) and ct.strip():
                corrected_text = ct.strip()

            # 한국어 검색 쿼리 추출
            rq = parsed.get("retrieval_query_ko")
            if isinstance(rq, str) and rq.strip():
                retrieval_query_ko = rq.strip()

            ranking = parsed.get("ranking", _DEFAULT_RANKING)
            # 1순위가 struct_db일 때만 place_category 적용, 허용 enum 외 값은 None
            raw_category = parsed.get("place_category")
            place_category = (
                raw_category.upper()
                if isinstance(raw_category, str) and raw_category.upper() in _ALLOWED_PLACE_CATEGORIES
                else None
            )

            # 특정 장소명 추출 (struct_db 1순위일 때만 유효)
            pnq = parsed.get("place_name_query")
            place_name_query = pnq.strip() if isinstance(pnq, str) and pnq.strip() else None

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
        except (json.JSONDecodeError, ValueError) as e:
            ranking = _DEFAULT_RANKING
            place_category = None
            place_name_query = None
            method = "llm_fallback_invalid_json"
            error = str(e)
        except (APIConnectionError, APITimeoutError) as e:
            ranking = _DEFAULT_RANKING
            place_category = None
            place_name_query = None
            method = "llm_fallback_connection_error"
            error = str(e)
        except RateLimitError as e:
            ranking = _DEFAULT_RANKING
            place_category = None
            place_name_query = None
            method = "llm_fallback_rate_limit"
            error = str(e)
        except APIError as e:
            ranking = _DEFAULT_RANKING
            place_category = None
            place_name_query = None
            method = "llm_fallback_api_error"
            error = f"{e.__class__.__name__}: {e}"

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("intent_gate")
        trace["_flow"] = flow
        trace["intent_gate"] = {
            "original_text": text,
            "corrected_text": corrected_text,
            "stt_corrected": corrected_text != text,
            "stt_language": stt_language,
            "actual_language": actual_language,
            "language_corrected": actual_language != stt_language,
            "retrieval_query_ko": retrieval_query_ko,
            "ranking": ranking,
            "place_category": place_category,
            "place_name_query": place_name_query,
            "method": method,
        }
        if error:
            trace["intent_gate"]["error"] = error

        # 상위 N개 의도만 시도
        ranking = ranking[:_MAX_INTENTS]

        # 1순위가 struct_db가 아니면 place_category, place_name_query 무효화
        if ranking[0] != "struct_db":
            place_category = None
            place_name_query = None

        return {
            "intent_ranking": ranking,
            "current_intent_index": 0,
            "place_category": place_category,
            "place_name_query": place_name_query,
            "normalized_text": corrected_text,
            "retrieval_query_ko": retrieval_query_ko,
            "user_language": actual_language,
            "answer_language": actual_language,  # user_language와 동기화
            "trace": trace,
        }

    return intent_gate_node
