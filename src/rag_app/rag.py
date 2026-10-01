# ============================================================
# 사용자의 질문과 관련된 문서를 FAISS에서 검색하고,
# 검색된 문서를 근거로 OpenAI LLM이 최종 답변을 생성합니다.
# 답변의 출처와 인용 번호가 올바른지도 함께 확인합니다.
# ============================================================

import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import config
from indexer import get_store
from prompts import PROMPTS


# ============================================================
# 1. 기본 환경 준비
# ============================================================

# .env 파일의 OPENAI_API_KEY를 환경변수로 불러옵니다.
load_dotenv()


# FAISS 벡터 저장소를 준비합니다.
store = get_store()


# OpenAI LLM을 준비합니다.
llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE
)


# 프롬프트 → LLM → 문자열 변환 순서로 실행 체인을 만듭니다.
chain = (
    PROMPTS[config.PROMPT_VER]
    | llm
    | StrOutputParser()
)


# ============================================================
# 2. 검색 함수
# ============================================================
# 질문과 관련된 문서를 벡터 저장소에서 검색합니다.
# MIN_SCORE 이상의 문서만 최종 검색 결과로 사용합니다.
# ============================================================

def _search(question, k=None):

    # 검색 개수가 지정되지 않으면 기본값을 사용합니다.
    if k is None:
        k = config.TOP_K

    # 질문과 유사한 문서를 검색합니다.
    results = store.similarity_search_with_relevance_scores(
        question,
        k=k
    )

    # 기준 점수를 통과한 문서만 저장합니다.
    docs = []

    for doc, score in results:

        if score >= config.MIN_SCORE:
            docs.append(doc)

    return docs


# ============================================================
# 3. 컨텍스트 생성 함수
# ============================================================
# 검색된 문서들을 LLM에게 전달하기 좋은 문자열로 만듭니다.
# 각 문서 앞에 번호, 파일명, 페이지 번호를 표시합니다.
# ============================================================

def _build_context(docs):

    # 문서별 문자열을 저장할 리스트입니다.
    parts = []

    # 검색된 문서를 하나씩 처리합니다.
    for number, doc in enumerate(docs, 1):

        # 원본 파일 이름을 가져옵니다.
        filename = doc.metadata.get(
            "filename",
            "unknown"
        )

        # 페이지 번호를 가져옵니다.
        page_no = doc.metadata.get("page_no")

        # page_no가 없으면 기본 page 값을 사용합니다.
        if page_no is None:
            page = doc.metadata.get("page", 0)
            page_no = page + 1

        # 출처 정보와 문서 내용을 하나의 문자열로 만듭니다.
        part = (
            f"[{number}] {filename} p.{page_no}\n"
            f"{doc.page_content}"
        )

        parts.append(part)

    # 문서 사이에 구분선을 넣어 하나의 문자열로 만듭니다.
    context = "\n\n---\n\n".join(parts)

    return context


# ============================================================
# 4. 질문 처리 함수
# ============================================================
# 질문을 받아 관련 문서를 검색하고,
# 검색 결과를 근거로 LLM에게 답변을 생성시킵니다.
# ============================================================

def ask(question: str, k: int = None) -> dict:

    # --------------------------------------------------------
    # 질문 입력 확인
    # --------------------------------------------------------

    # 질문이 없거나 공백만 있으면 종료합니다.
    if not question or not question.strip():

        return {
            "answer": "질문을 입력해주세요.",
            "sources": [],
            "ok": False
        }


    # --------------------------------------------------------
    # 관련 문서 검색
    # --------------------------------------------------------

    try:
        docs = _search(question, k)

    except Exception as error:

        print("[검색 오류]", error)

        return {
            "answer": config.MSG_ERROR,
            "sources": [],
            "ok": False
        }


    # --------------------------------------------------------
    # 검색 결과 확인
    # --------------------------------------------------------

    # 관련 문서를 찾지 못했다면 LLM을 호출하지 않습니다.
    if not docs:

        return {
            "answer": config.MSG_NO_DOC,
            "sources": [],
            "ok": True
        }


    # --------------------------------------------------------
    # LLM에게 전달할 근거 문서 만들기
    # --------------------------------------------------------

    context = _build_context(docs)


    # --------------------------------------------------------
    # LLM 답변 생성
    # --------------------------------------------------------

    try:

        answer = chain.invoke({
            "context": context,
            "question": question
        })

    except Exception as error:

        print("[생성 오류]", error)

        return {
            "answer": config.MSG_ERROR,
            "sources": [],
            "ok": False
        }


    # --------------------------------------------------------
    # 답변의 인용 번호 확인
    # --------------------------------------------------------

    # 답변에서 [1], [2] 같은 인용 번호를 찾습니다.
    found = re.findall(
        r"\[(\d+)\]",
        answer
    )

    # 찾은 문자열 번호를 정수로 변환합니다.
    citation_numbers = []

    for number in found:
        citation_numbers.append(int(number))


    # 인용 번호가 하나라도 있으면 True로 시작합니다.
    cited = False

    if len(citation_numbers) > 0:
        cited = True


    # 실제 문서 범위를 벗어난 인용이 있는지 확인합니다.
    for number in citation_numbers:

        if number < 1 or number > len(docs):
            cited = False


    # --------------------------------------------------------
    # 출처 정보 정리
    # --------------------------------------------------------

    sources = []

    # 검색에 사용된 문서의 출처를 하나씩 저장합니다.
    for doc in docs:

        # 파일 이름을 가져옵니다.
        filename = doc.metadata.get(
            "filename",
            "unknown"
        )

        # 페이지 번호를 가져옵니다.
        page_no = doc.metadata.get("page_no")

        # page_no가 없으면 기본 page 값을 사용합니다.
        if page_no is None:
            page = doc.metadata.get("page", 0)
            page_no = page + 1

        # 출처 정보를 저장합니다.
        source = {
            "file": filename,
            "page": page_no
        }

        sources.append(source)


    # --------------------------------------------------------
    # 최종 결과 반환
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
        "cited": cited,
        "ok": True
    }