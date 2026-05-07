from __future__ import annotations

# from app.core.services.stt_google import GoogleSTT
from app.core.services.stt_google_v2 import GoogleSTTV2 as GoogleSTT
from app.graph.state import GraphState


def make_stt_node(stt: GoogleSTT):
    """STT 서비스를 주입받아 노드 함수를 반환하는 팩토리."""

    def stt_node(state: GraphState) -> dict:
        audio: bytes = state.get("audio", b"")

        result = stt.transcribe(audio)

        trace = dict(state.get("trace") or {})
        flow = list(trace.get("_flow") or [])
        flow.append("stt")
        trace["_flow"] = flow
        trace["stt"] = {
            "transcript": result.transcript,
            "stt_detected_language": result.language_code,  # STT 모델이 감지한 언어 (참고용)
            "stt_language_code": state.get("stt_language_code", ""),  # 실제 사용한 BCP-47 코드
            "confidence": result.confidence,
        }

        return {
            "transcript": result.transcript,
<<<<<<< HEAD
            "language_code": result.language_code,
            "detected_language_code": result.language_code,
            "user_language": result.language_code,  # 원언어 보존 (끝까지 유지)
=======
            "language_code": result.language_code,  # STT 감지 언어 (참고용)
            # user_language / answer_language 는 LanguageProfile에서 설정된 값을 유지
            # (stt_node에서 덮어쓰지 않음)
>>>>>>> 65f97c2 (lang code 관련 수정중 (#149))
            "trace": trace,
        }

    return stt_node
