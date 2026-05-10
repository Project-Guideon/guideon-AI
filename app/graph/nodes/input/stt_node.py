from __future__ import annotations

from app.core.services.stt_google_v2 import GoogleSTTV2 as GoogleSTT
from app.graph.state import GraphState


def make_stt_node(stt: GoogleSTT):
    """STT 서비스를 주입받아 노드 함수를 반환하는 팩토리."""

    def stt_node(state: GraphState) -> dict:
        audio: bytes = state.get("audio", b"")
        primary_language = state.get("stt_language_code") or None

        result = stt.transcribe(audio, primary_language=primary_language)

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("stt")
        trace["_flow"] = flow
        trace["stt"] = {
            "transcript": result.transcript,
            "stt_detected_language": result.language_code,  # STT 모델이 감지한 언어 (참고용)
            "stt_language_code": state.get("stt_language_code", ""),
            "confidence": result.confidence,
        }

        return {
            "transcript": result.transcript,
            "language_code": result.language_code,  # STT 감지 언어 (참고용)
            # user_language / answer_language 는 LanguageProfile에서 설정된 값을 유지
            "trace": trace,
        }

    return stt_node
