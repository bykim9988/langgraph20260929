from langchain_community.document_loaders import PyPDFLoader
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)  # 지금 현재 디렉토리
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "06")
sys.path.insert(0, INGEST_DIR)

from ingest import load_documents   # 06\ingest.py의 함수

docs = load_documents("../../data/manual.pdf")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 50,
    separators=["\n\n", "\n", ".", " ", ""], # 문단 -> 빈 줄 => 마침표 -> 빈 칸 -> 그냥 잘자라 로 우선순위 지정
    length_function = len,

)

chunks = splitter.split_documents(docs) # -> List
print(len(chunks))
print(chunks[0].page_content)
print('---')
print(chunks[0].metadata)
print(chunks[0].page_content[-80:])
print('===')
print(chunks[1].page_content)
print(chunks[1].metadata)
print(chunks[1].page_content[:80])
