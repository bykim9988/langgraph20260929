from langchain_community.document_loaders import PyPDFLoader
import os
import sys

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CURRENT_DIR = os.path.dirname(__file__)
INGEST_DIR = os.path.join(CURRENT_DIR, "..", "06")
sys.path.insert(0, INGEST_DIR)

from ingest import load_documents

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

def prepare_chunks(path):

    docs = load_documents(path)

    print(docs[0].metadata)

    print(docs[147].page_content)
    print(f"page: {docs[147].metadata['page_no']}")
    print('---')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function = len,
    )

    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):   # 각 청크에 고유번호 붙이기
        chunk.metadata["chunk_id"] = index

    print(f"청크 {len(chunks)}개")

    return chunks

if __name__ == "__main__":

    chunks = prepare_chunks("../../data/manual.pdf")

    for chunk in chunks[-2:]:
        print(chunk.page_content)
        print(chunk.metadata["chunk_id"], f": 총 {len(chunk.page_content)}자")
        print("---")

    
