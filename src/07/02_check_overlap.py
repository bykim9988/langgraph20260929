from langchain_community.document_loaders import PyPDFLoader
import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CURRENT_DIR = os.path.dirname(__file__)  # 지금 현재 디렉토리
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "06")
sys.path.insert(0, INGEST_DIR)

from ingest import load_documents   # 06\ingest.py의 함수

docs = load_documents("../../data/manual.pdf")

# pypdfloader는 페이지 경계면에서는 overlap 되지 않음

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 300,
    # separators=["\n\n", "\n", ".", " ", ""], # 문단 -> 빈 줄 => 마침표 -> 빈 칸 -> 그냥 잘자라 로 우선순위 지정
    length_function = len,

)

full_text = ""
for doc in docs:
    full_text += doc.page_content + "\n\n"  # 페이지 뒤에 "\n\n"을 붙였기 때문에 무조건 페이지에서 split됨.

merged_doc = Document(page_content = full_text)

chunks = splitter.split_documents([merged_doc])   # -> List

def show_boundary(chunks, index=0):
    print(chunks[index].page_content[ -80:])
    print(f"(총 {len(chunks[index].page_content)}자)")
    print(chunks[index].page_content)
    print("---")
    print(chunks[index+1].page_content[ : 80])
    print(chunks[index+1].metadata)

show_boundary(chunks, 0)


