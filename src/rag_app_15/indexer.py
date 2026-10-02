# ============================================================
# PDF 문서를 읽고 정리한 뒤 FAISS 벡터 인덱스를 생성합니다.
# 저장된 인덱스가 있으면 새로 만들지 않고 기존 인덱스를 불러옵니다.
# ============================================================

import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# PDF에서 반복적으로 제거할 불필요한 문구
NOISE = ["㈜한국주식회사 대외비"]


# ============================================================
# PDF 문서를 읽고 텍스트와 메타데이터를 정리합니다.
# ============================================================
def _load():

    # PDF 파일의 모든 페이지를 읽습니다.
    loader = PyPDFLoader(str(config.DOC_PATH))
    docs = loader.load()

    # 각 페이지를 하나씩 정리합니다.
    for doc in docs:

        # 페이지의 텍스트를 가져옵니다.
        text = doc.page_content

        # 불필요한 문구를 제거합니다.
        for noise in NOISE:
            text = text.replace(noise, "")

        # 연속된 빈 줄을 최대 두 줄로 정리합니다.
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 정리된 텍스트를 다시 저장합니다.
        doc.page_content = text.strip()

        # 원본 파일 이름을 metadata에 저장합니다.
        doc.metadata["filename"] = config.DOC_PATH.name

        # 사람이 보기 편하도록 페이지 번호를 1부터 시작합니다.
        page = doc.metadata.get("page", 0)
        doc.metadata["page_no"] = page + 1

    # 모든 페이지 처리가 끝난 뒤 반환합니다.
    return docs


# ============================================================
# PDF 페이지를 검색에 적합한 작은 조각(chunk)으로 나눕니다.
# ============================================================
def _split(docs):

    # 문서를 나누는 도구를 만듭니다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # 전체 문서를 작은 조각으로 나눕니다.
    chunks = splitter.split_documents(docs)

    # 각 조각에 고유 번호를 붙입니다.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


# ============================================================
# FAISS 벡터 저장소를 준비합니다.
# 기존 인덱스가 있으면 로드하고, 없으면 새로 생성합니다.
# ============================================================
def get_store(rebuild=False):

    # OpenAI 임베딩 모델을 준비합니다.
    embeddings = OpenAIEmbeddings(
        model=config.EMBED_MODEL
    )

    # 저장된 인덱스가 있고 강제 재생성이 아니면 불러옵니다.
    if os.path.exists(config.INDEX_PATH) and not rebuild:

        store = FAISS.load_local(
            config.INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return store

    # PDF 문서를 읽습니다.
    docs = _load()

    # 문서를 작은 조각으로 나눕니다.
    chunks = _split(docs)

    # 각 조각을 임베딩하여 FAISS 인덱스를 만듭니다.
    store = FAISS.from_documents(
        chunks,
        embeddings
    )

    # 생성한 인덱스를 디스크에 저장합니다.
    store.save_local(config.INDEX_PATH)

    print(f"✓ 인덱스 생성 완료 (조각 {len(chunks)}개)")

    return store