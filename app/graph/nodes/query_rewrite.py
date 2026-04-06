from __future__ import annotations

from app.core.services.llm_openai import OpenAILLM
from app.graph.state import GraphState


def make_query_rewrite_node(llm: OpenAILLM):
    """RAG 검색용 쿼리 재작성 노드 팩토리.

    - KO 경로  : normalized_text → retrieval_query_ko 생성
    - Foreign  : translate_node 가 만든 retrieval_query_ko 를 추가로 다듬음
    """

    def query_rewrite_node(state: GraphState) -> dict:
        # Foreign 경로는 translate_node 가 이미 retrieval_query_ko 를 세팅함
        # KO 경로는 아직 없으므로 normalized_text 를 기본값으로 사용
        base_query: str = state.get("retrieval_query_ko") or state.get("normalized_text", "")
        site_id: int = state.get("site_id", 1)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer for a Korean tourism RAG system.\n"
                    "Rewrite the input into a SHORT, keyword-focused Korean search query.\n"
                    "Rules:\n"
                    f"  - Site context: site_id={site_id} (tourism/heritage site)\n"
                    "  - Output ONLY the rewritten Korean query, no explanation\n"
                    "  - Keep it concise (5-15 words)\n"
                    "  - Focus on nouns and key terms, remove filler words\n"
                    "  - Preserve proper nouns (place names, monument names, person names)\n"
                    "  - Do NOT answer the question"
                ),
            },
            {
                "role": "user",
                "content": f"Rewrite for RAG search:\n{base_query}",
            },
        ]

        try:
            rewritten = llm.chat(messages, max_tokens=60).strip()
        except Exception:
            rewritten = base_query  # 실패 시 원문 그대로

        trace = dict(state.get("trace") or {})
        trace["query_rewrite"] = {
            "base_query": base_query,
            "rewritten": rewritten,
        }

        return {"retrieval_query_ko": rewritten, "trace": trace}

    return query_rewrite_node
