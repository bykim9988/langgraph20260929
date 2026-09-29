from langchain_community.document_loaders import PyPDFLoader
import os
import re

# loader = PyPDFLoader("../../data/notice.pdf",)
# loader = PyPDFLoader("../../data/scan_image.pdf",)

NOISE = ["대외비"]

# documents = loader.load()  # list of Document objects

def load_documents(path):

    loader = PyPDFLoader(path)
    docs = loader.load()  # list of Document objects, 메모리에 로딩

    empty_pages = []
    for d in docs:  # docs is a list of Document objects
        text = d.page_content

        for noise in NOISE:     # 모든 노이즈 단어 제거
            text = text.replace(noise, "")    
        re.sub(r"\n{3,}", "\n\n", text)  # 연속된 줄 바꿈은 최대 2번으로 줄임

        d.page_content = text.strip()  # remove leading/trailing whitespace

        # 메타 데이터
        d.metadata["filename"] = os.path.basename(path)
        d.metadata["page_no"] = d.metadata.get("page", 0) + 1 # 페이지의 시작은 1부터로..

    empty_pages = []
    for d in docs:  # docs is a list of Document objects
        text = d.page_content.strip()  # remove leading/trailing whitespace
        if len(text) < 10:
            page = d.metadata.get("page", "?")
            empty_pages.append(page)

    if empty_pages:
        print("스캔 pdf 일 수 있습니다.")

    total_len = 0
    for d in docs:
        total_len += len(d.page_content)

    print(f"{len(docs)}쪽 로딩 완료 (총 {total_len}자)")
    return docs


if __name__ == "__main__":
    docs = load_documents("../../data/manual.pdf")
    print( docs[1].page_content )
