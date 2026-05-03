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
            "language_code": result.language_code,
            "confidence": result.confidence,
        }

        return {
            "transcript": result.transcript,
            "language_code": result.language_code,
            "detected_language_code": result.language_code,
            "user_language": result.language_code,  # 원언어 보존 (끝까지 유지)
            "trace": trace,
        }

    return stt_node
