from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../data/notice.txt", encoding="utf-8")
documents = loader.load()  # list of Document objects

print(f"Loaded {len(documents)} documents.")

for i, doc in enumerate(documents):
    print(f"Document {i + 1}:")
    print(doc.page_content)
    print("---")
    print(f"Metadata: {doc.metadata}")
    print("===")