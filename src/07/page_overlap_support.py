# 필수 설치: pip install pymupdf4llm langchain-text-splitters
import re
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdf_with_perfect_overlap_and_metadata(pdf_path, chunk_size=1200, chunk_overlap=200):
    # 1. page_chunks=True 옵션으로 페이지별 마크다운과 메타데이터를 분리해서 가져옵니다.
    # 이 옵션을 쓰면 표 서식이 마크다운 형태로 완벽히 유지됩니다.
    pages_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True) #
    
    # 2. 각 페이지의 텍스트에 특수 페이지 태그를 붙여서 하나의 거대한 문자열로 합칩니다.
    # 이렇게 해야 텍스트 분할기가 페이지 경계를 무시하고 오버랩을 정상 작동시킵니다.
    full_markdown_text = ""
    for page in pages_data:
        page_num = page["metadata"]["page_number"] # 0부터 시작하는 페이지 번호
        page_text = page["text"]
        
        # 페이지 시작 지점에 고유 식별자 태그 삽입
        full_markdown_text += f"\n<!-- PAGE_START_{page_num} -->\n{page_text}\n<!-- PAGE_END_{page_num} -->\n"

    # 3. 텍스트 분할기로 전체 문자열을 자릅니다 (페이지 경계면 오버랩 완벽 작동)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_text(full_markdown_text)
    
    final_documents = []
    
    # 4. 분할된 각 청크 내부의 태그를 추적하여 해당 청크가 어떤 페이지들에 걸쳐 있는지 역추적합니다.
    for chunk in text_chunks:
        # 청크 안에 포함된 모든 페이지 번호 패턴 추출
        start_pages = re.findall(r"<!-- PAGE_START_(\d+) -->", chunk)
        end_pages = re.findall(r"<!-- PAGE_END_(\d+) -->", chunk)
        
        # 정규식 태그 자체는 LLM에게 방해가 되므로 깔끔하게 지워줍니다.
        clean_content = re.sub(r"<!-- (PAGE_START|PAGE_END)_\d+ -->\n?", "", chunk).strip()
        
        # 만약 청크 내부에 태그가 없다면, 전체 텍스트 내에서 해당 청크의 위치를 기반으로 페이지를 계산합니다.
        if not start_pages and not end_pages:
            # 이전 청크의 페이지 정보를 기반으로 유추하거나 가장 가까운 페이지 매핑 (예외 처리)
            start_pos = full_markdown_text.find(chunk)
            # 현재 청크 시작점 이전의 가장 최근 시작 태그를 찾음
            nearest_start = re.findall(r"<!-- PAGE_START_(\d+) -->", full_markdown_text[:start_pos])
            associated_pages = [int(nearest_start[-1])] if nearest_start else [0]
        else:
            # 태그가 존재한다면 찾아낸 페이지 번호들을 취합합니다. (오버랩 구간은 2개 이상의 페이지 번호가 잡힙니다)
            all_found_pages = list(map(int, start_pages + end_pages))
            
            # 텍스트 흐름상 누락된 중간 페이지가 있다면 보정 (ex: 1페이지 시작 태그와 3페이지 끝 태그만 잡힌 경우 2페이지 포함)
            min_p, max_p = min(all_found_pages), max(all_found_pages)
            associated_pages = list(range(min_p, max_p + 1))
            
        # 사람이 읽기 좋은 페이지 번호 형태로 변환 (1부터 시작하는 페이지 표기법, 예: "1, 2")
        page_string = ", ".join(map(str, [p + 1 for p in sorted(associated_pages)]))
        
        # 5. LangChain 표준 Document 객체 생성 및 메타데이터 주입
        doc = Document(
            page_content=clean_content,
            metadata={
                "source": pdf_path,
                "pages": page_string,  # 걸쳐 있는 모든 페이지 번호 기록 (예: "3" 또는 "4, 5")
                "has_table": "|" in clean_content # 청크 내에 표가 포함되어 있는지 여부 체크
            }
        )
        final_documents.append(doc)
        
    return final_documents

# --- 사용 예시 ---
pdf_file = "your_complex_document.pdf"
documents = load_pdf_with_perfect_overlap_and_metadata(pdf_file, chunk_size=1000, chunk_overlap=200)

# 결과 확인
print(f"생성된 총 청크 개수: {len(documents)}")
print("\n=== 오버랩 경계면 청크 및 메타데이터 확인 ===")
for i, d in enumerate(documents[:3]):
    print(f"[{i+1}번째 청크] 출처 페이지: {d.metadata['pages']}쪽 / 표 포함 여부: {d.metadata['has_table']}")
    print(d.page_content[:200] + "...\n" + "-"*50)
