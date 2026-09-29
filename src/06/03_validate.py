from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("../../data/notice.pdf",)
loader = PyPDFLoader("../../data/scan_image.pdf",)
documents = loader.load()  # list of Document objects

def validate(docs):

    empty_pages = []
    for d in docs:  # docs is a list of Document objects
        text = d.page_content.strip()  # remove leading/trailing whitespace
        if len(text) < 10:
            page = d.metadata.get("page", "?")
            empty_pages.append(page)

    if empty_pages:
        print("스캔 pdf 일 수 있습니다.")

validate(documents)