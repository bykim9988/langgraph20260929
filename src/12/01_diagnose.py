import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CH10_DIR = os.path.join(CURRENT_DIR, "..", "10")
CH11_DIR = os.path.join(CURRENT_DIR, "..", "11")

sys.path.insert(0, CH10_DIR)
sys.path.insert(0, CH11_DIR)

from indexer import get_store
from retriever import build_context, MIN_SCORE

load_dotenv()

store = get_store()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature = 0
)

def diagnose(question, k=3):

    print("=" * 60)
    print("Q:", question)

    pairs = store.similarity_search_with_relevance_scores(
        question,
        k=k
    )

    if not pairs:
        print("→ 【검색 실패】 결과가 0건입니다")
        return

    passed = []

    for document, score in pairs:
        if score >= MIN_SCORE:
            passed.append((document, score))

    print(
        f"검색 {len(pairs)}개 / "
        f"기준 통과 {len(passed)}개"
    )

    print("-" * 60)
    for index, (document, score) in enumerate(pairs, start=1):
        if score >= MIN_SCORE:
            mark = "✓"
        else:
            mark = "✗"
        text = document.page_content[:70]
        text = text.replace("\n", " ")
        page_no = document.metadata["page_no"]
        print(
            f"{mark} [{index}] "
            f"{score:.3f} "
            f"p.{page_no} "
            f"{text}..."
        )

    docs = []

    for document, score in passed:
        docs.append(document)
    
    context = build_context(docs)
    prompt = (
        "아래 자료만 근거로 답하세요.\n"
        "자료에 없으면 "
        "'자료에서 확인할 수 없습니다'라고 답하세요.\n\n"
        f"[자료]\n{context}\n\n"
        f"[질문] {question}"
    )
    response = llm.invoke(prompt)
    answer = response.content
    print("\nA:", answer)

    print("\n👉 판단: 위 조각들 안에 정답이 있었는가?")
    print("   있는데 답이 틀렸다면 → 생성 문제 (13차시)")
    print("   없다면              → 검색 문제 (7·11차시)")

    print("=" * 60)


if __name__ == "__main__":
    diagnose("환불은 몇일인가요?")

