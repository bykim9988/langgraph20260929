import streamlit as st

from rag import ask

@st.cache_resource
def load_rag():

    # rag.py의 ask 함수를 가져옵니다.
    from rag import ask

    # 가져온 ask 함수를 반환합니다.
    return ask


# 캐시된 ask 함수를 가져옵니다.
ask = load_rag()

st.set_page_config(
     page_title="문서 Q&A",
     page_icon="■"
)

st.title("■ 문서 기반 질의 응답")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input(
    "궁금한 것을 물어보세요"
)

if question:

    # 답변을 만드는 동안 로딩 메시지를 표시합니다.
    with st.spinner("자료를 찾는 중..."):

        # RAG 시스템에 질문을 전달합니다.
        result = ask(question)

    # 질문과 결과를 대화 기록에 저장합니다.
    st.session_state.history.append(
        (question, result)
    )


for question, result in st.session_state.history:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        st.write(result["answer"])

        if result["sources"]:
            pages = []

            for source in result["sources"]:
                page = source["page"]

                if page not in pages:
                    pages.append(page)

            pages.sort()

            # 숫자 페이지를 문자열로 변환합니다.
            page_texts = []

            for page in pages:
                page_texts.append(str(page))


            # 1, 2, 3 형태의 문자열로 만듭니다.
            page_text = ", ".join(page_texts)


            # 첫 번째 검색 결과에서 파일 이름을 가져옵니다.
            filename = result["sources"][0]["file"]


            # 참고 문서와 페이지 정보를 출력합니다.
            st.caption(
                f"※ 참고: {filename} "
                f"{page_text}페이지"
            )

