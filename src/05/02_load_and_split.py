from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("../../data/notice.txt", encoding="utf-8")
documents = loader.load()  # list of Document objects

# 문서를 청크로 나누기
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,  # 각 청크의 최대 길이
    chunk_overlap=30,  # 청크 간의 겹침 길이
)

chunks =splitter.split_documents(documents)  # 문서를 청크로 나누기

print(f"{len(chunks)} chunks created from {len(documents)} documents.")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk.page_content)
    print("---")
    print(f"Metadata: {chunk.metadata}")
    print("===")
    
print(f"Loaded {len(documents)} documents.")

for i, doc in enumerate(documents):
    print(f"Document {i + 1}:")
    print(doc.page_content)
    print("---")
    print(f"Metadata: {doc.metadata}")
    print("===")