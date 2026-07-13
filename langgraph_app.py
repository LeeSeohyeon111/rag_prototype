import os
import time
import json
import operator
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv()

# ── 로깅 시스템 ──────────────────────────────────────────
def log_event(event_type: str, data: dict):
    """
    이벤트 로깅 함수
    - event_type: 'query', 'error', 'feedback' 등
    - data: 로그에 담을 데이터
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        **data
    }

    # session_state에 로그 누적
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(log_entry)

    # 파일에도 저장 (append 모드)
    try:
        with open("app_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 파일 저장 실패해도 앱은 계속 동작

# ── 1. State 정의 ──────────────────────────────────────────
class TechGPTState(TypedDict):
    question: str
    intents:  List[str]
    context:  str
    answers:  Annotated[List[str], operator.add]
    answer:   str

# ── 2. LLM (Multi-Model 라우팅 적용) ────────────────────────
cheap_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
smart_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# ── 3. 노드 함수 정의 ──────────────────────────────────────
def intent_analyzer_node(state: TechGPTState) -> TechGPTState:
    """① Intent Analyzer Node (GPT-4o-mini)"""
    prompt = ChatPromptTemplate.from_template("""
    #역할
    당신은 사용자 질문의 의도를 정확히 분류하는 전문 분석가입니다.

    #명령문
    아래 질문을 분석하여 의도를 분류하세요.

    #제약조건
    - search / summary / general 중 해당하는 것을 모두 골라 쉼표로 구분하세요
    - 해당하는 의도가 여러 개면 전부 출력하세요

    #예시 (Few Shot)
    Q: 이 문서 요약해줘 → summary
    Q: 사업 목적이 뭐야? → search
    Q: 안녕 → general
    Q: 요약하고 사업 목적도 알려줘 → summary,search

    #입력문
    질문: {question}

    #출력형식
    의도 (쉼표 구분):""")

    chain = prompt | cheap_llm | StrOutputParser()
    raw = chain.invoke({"question": state["question"]}).strip().lower()

    valid   = {"search", "summary", "general"}
    intents = [i.strip() for i in raw.split(",") if i.strip() in valid]
    if not intents:
        intents = ["general"]

    # 로깅: intent 분류 결과
    log_event("intent_classified", {
        "question": state["question"],
        "intents": intents,
        "raw_output": raw
    })

    return {"intents": intents}

def summary_node(state: TechGPTState) -> TechGPTState:
    """②-A Summary Node (GPT-4o / 병렬 실행)"""
    # 문서가 업로드되지 않은 경우 안전 처리
    if "split_docs" not in st.session_state:
        return {"answers": ["⚠️ 요약을 위해서는 먼저 PDF 문서를 업로드해주세요."]}

    all_docs = st.session_state.split_docs
    BATCH_SIZE = 5
    batches = [
        all_docs[i:i + BATCH_SIZE]
        for i in range(0, len(all_docs), BATCH_SIZE)
    ]

    map_prompt = ChatPromptTemplate.from_template("""
    문서 조각의 핵심 내용을 통합하여 요약하세요. 문서에 없는 내용은 추가하지 마세요.
    [문서 조각들]
    {chunks}
    """)
    map_chain = map_prompt | smart_llm | StrOutputParser()

    batch_summaries = []
    for batch in batches:
        chunks_text = "\n\n".join([doc.page_content for doc in batch])
        batch_summaries.append(map_chain.invoke({"chunks": chunks_text}))

    combined = "\n\n".join(batch_summaries)

    reduce_prompt = ChatPromptTemplate.from_template("""
    아래 부분 요약들을 바탕으로 전체 문서를 최종 요약하세요.
    [부분 요약]
    {combined}

    #출력형식
    **📌 핵심 목적** (한 줄)
    **📋 주요 내용** (불릿 포인트)
    **💡 핵심 키워드** (쉼표 구분)
    """)
    reduce_chain = reduce_prompt | smart_llm | StrOutputParser()
    summary_result = reduce_chain.invoke({"combined": combined})

    return {"answers": [summary_result]}

def search_node(state: TechGPTState) -> TechGPTState:
    """②-B Search Node (GPT-4o / 병렬 실행)"""
    # 문서가 업로드되지 않은 경우 안전 처리
    if "bm25" not in st.session_state or "vectorstore" not in st.session_state:
        return {"answers": ["⚠️ 검색을 위해서는 먼저 PDF 문서를 업로드해주세요."]}

    bm25 = st.session_state.bm25
    vector = st.session_state.vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    ensemble = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])

    multi_query = MultiQueryRetriever.from_llm(retriever=ensemble, llm=smart_llm)

    docs = multi_query.invoke(state["question"])
    context = "\n\n".join([
        f"[출처: {doc.metadata.get('source', '알 수 없음')} | 페이지: {doc.metadata.get('page', '?')}페이지]\n{doc.page_content}"
        for doc in docs
    ])

    # 로깅: 검색된 문서 개수
    log_event("retrieval", {
        "question": state["question"],
        "docs_retrieved": len(docs),
        "context_length": len(context)
    })

    answer_prompt = ChatPromptTemplate.from_template("""
    아래 문서를 기반으로 질문에 답하세요.
    [참고 문서]
    {context}
    [질문]
    {question}

    #출력형식
    ✅ **답변** (2~3줄)
    📎 **근거** (문서에서 인용)
    """)
    answer_chain = answer_prompt | smart_llm | StrOutputParser()
    answer_result = answer_chain.invoke({"context": context, "question": state["question"]})

    return {"context": context, "answers": [answer_result]}

def general_node(state: TechGPTState) -> TechGPTState:
    """②-C General Node (GPT-4o-mini / 병렬 실행)"""
    general_prompt = ChatPromptTemplate.from_template("""
    사용자의 말에 친절하게 응답하고, 자연스럽게 문서 기반 질문을 유도하는 말로 마무리하세요.
    사용자: {question}
    AI: 안녕하세요! 저는 기술 정보 AI입니다. (이어서 2줄 이내로 완성)
    """)
    general_chain = general_prompt | cheap_llm | StrOutputParser()
    general_result = general_chain.invoke({"question": state["question"]})

    return {"answers": [general_result]}

def merge_node(state: TechGPTState) -> TechGPTState:
    """③ Merge Node (GPT-4o)"""
    answers = state["answers"]
    if len(answers) == 1:
        return {"answer": answers[0]}

    merge_prompt = ChatPromptTemplate.from_template("""
    아래 여러 개의 답변을 하나의 자연스러운 응답으로 통합하세요.
    출처 정보는 유지하고, 요약을 먼저 제시한 뒤 구체적 답변을 이어가세요.
    {answers}
    """)
    merge_chain = merge_prompt | smart_llm | StrOutputParser()
    answer = merge_chain.invoke({"answers": "\n\n---\n\n".join(answers)})
    return {"answer": answer}

# ── 4. 라우팅 함수 ─────────────────────────────────────────
def route_to_agents(state: TechGPTState) -> List[str]:
    intents = state["intents"]
    destinations = []
    if "summary" in intents: destinations.append("summary_node")
    if "search" in intents:  destinations.append("search_node")
    if "general" in intents: destinations.append("general_node")
    return destinations

# ── 5. 그래프 구성 ─────────────────────────────────────────
def build_graph():
    graph = StateGraph(TechGPTState)

    graph.add_node("intent_analyzer", intent_analyzer_node)
    graph.add_node("summary_node",    summary_node)
    graph.add_node("search_node",     search_node)
    graph.add_node("general_node",    general_node)
    graph.add_node("merge",           merge_node)

    graph.set_entry_point("intent_analyzer")

    graph.add_conditional_edges(
        "intent_analyzer",
        route_to_agents,
        {"summary_node": "summary_node", "search_node": "search_node", "general_node": "general_node"}
    )

    graph.add_edge("summary_node", "merge")
    graph.add_edge("search_node",  "merge")
    graph.add_edge("general_node", "merge")
    graph.add_edge("merge", END)

    return graph.compile()

# ── 6. 다중 PDF 처리 함수 ───────────────────────────────────────
def process_pdfs(pdf_paths: List[str]):
    try:
        all_docs = []
        for path in pdf_paths:
            loader = PyMuPDFLoader(path)
            all_docs.extend(loader.load())

        if not all_docs:
            raise ValueError("문서에서 텍스트를 추출할 수 없습니다.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=400, separators=["\n\n", "\n", "│", "─", " ", ""]
        )
        split_docs = splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        bm25 = BM25Retriever.from_documents(split_docs)
        bm25.k = 5

        st.session_state.split_docs = split_docs
        st.session_state.bm25 = bm25

        # 로깅: PDF 처리 성공
        log_event("pdf_processed", {
            "num_files": len(pdf_paths),
            "num_chunks": len(split_docs)
        })

        return vectorstore
    except Exception as e:
        # 로깅: 오류
        log_event("error", {
            "location": "process_pdfs",
            "error": str(e)
        })
        st.error(f"다중 문서 병합 및 인덱싱 오류: {e}")
        return None

# ── 7. Streamlit UI ───────────────
st.set_page_config(page_title="Tech-GPT Enterprise", layout="wide")

with st.sidebar:
    st.header("⚡ LLM Routing System")
    st.caption("비용 최적화 및 응답 속도 향상을 위해 작업 난이도에 따라 모델이 동적으로 분기됩니다.")
    st.markdown("- 🟢 `GPT-4o-mini`: 의도 분석, 일상 대화\n- 🟣 `GPT-4o`: 심층 RAG 검색, 문서 요약")

    st.divider()

    # 📊 모니터링 대시보드
    st.header("📊 모니터링")
    if "logs" in st.session_state and st.session_state.logs:
        total_queries = sum(1 for log in st.session_state.logs if log["event_type"] == "query_complete")
        total_errors = sum(1 for log in st.session_state.logs if log["event_type"] == "error")

        st.metric("총 질문 수", total_queries)
        st.metric("오류 발생", total_errors)

        # 평균 응답 시간
        response_times = [
            log.get("elapsed_time", 0)
            for log in st.session_state.logs
            if log["event_type"] == "query_complete"
        ]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            st.metric("평균 응답 시간", f"{avg_time:.2f}초")

        # 총 토큰 사용량
        total_tokens = sum(
            log.get("total_tokens", 0)
            for log in st.session_state.logs
            if log["event_type"] == "query_complete"
        )
        st.metric("총 토큰 사용", f"{total_tokens:,}")

        # 총 비용 (달러)
        total_cost = sum(
            log.get("cost", 0)
            for log in st.session_state.logs
            if log["event_type"] == "query_complete"
        )
        st.metric("총 비용", f"${total_cost:.4f}")

        # 로그 다운로드
        st.divider()
        if st.button("📥 로그 다운로드"):
            log_json = json.dumps(st.session_state.logs, ensure_ascii=False, indent=2)
            st.download_button(
                label="JSON 다운로드",
                data=log_json,
                file_name=f"tech_gpt_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("아직 로그가 없습니다.")

st.title("Tech-GPT : Enterprise AI Assistant")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📄 다중 문서 업로드")

    uploaded_files = st.file_uploader(
        "분석할 모든 PDF 문서들을 한 번에 업로드하세요.",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        import tempfile

        current_files_id = "-".join(sorted([f"{f.name}_{f.size}" for f in uploaded_files]))

        if st.session_state.get("processed_files_id") != current_files_id:
            if "vectorstore" in st.session_state:
                del st.session_state["vectorstore"]
            if "messages" in st.session_state:
                st.session_state.messages = []

        if "vectorstore" not in st.session_state:
            tmp_paths = []

            try:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_paths.append(tmp_file.name)

                with st.spinner(f"🚀 {len(uploaded_files)}개의 문서를 병합 및 벡터화 중..."):
                    vs = process_pdfs(tmp_paths)
                    if vs:
                        st.session_state.vectorstore = vs
                        st.session_state.processed_files_id = current_files_id
                        st.success(f"성공! {len(uploaded_files)}개 문서 교차 검색 준비 완료.")

            finally:
                for path in tmp_paths:
                    if os.path.exists(path):
                        os.unlink(path)

with col2:
    st.markdown("### 💬 대화 콘솔")
    if "vectorstore" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container(height=500)

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if user_input := st.chat_input("문서 내용에 대해 무엇이든 물어보세요."):
            st.session_state.messages.append({"role": "user", "content": user_input})

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("에이전트 라우팅 및 분석 중..."):
                        # 응답 시간 & 토큰 사용량 측정
                        start_time = time.time()

                        try:
                            with get_openai_callback() as cb:
                                response = st.session_state.graph.invoke({"question": user_input})
                                final_answer = response.get("answer", "답변을 생성하지 못했습니다.")

                            elapsed = time.time() - start_time

                            # 로깅: 질문 완료
                            log_event("query_complete", {
                                "question": user_input,
                                "answer_length": len(final_answer),
                                "elapsed_time": round(elapsed, 2),
                                "total_tokens": cb.total_tokens,
                                "prompt_tokens": cb.prompt_tokens,
                                "completion_tokens": cb.completion_tokens,
                                "cost": round(cb.total_cost, 6)
                            })

                            st.markdown(final_answer)

                            # 응답 메타데이터 표시 (작게)
                            st.caption(f"⏱️ {elapsed:.2f}초 | 🔤 {cb.total_tokens:,} 토큰 | 💰 ${cb.total_cost:.4f}")

                            st.session_state.messages.append({"role": "assistant", "content": final_answer})

                        except Exception as e:
                            # 로깅: 오류
                            log_event("error", {
                                "location": "chat_invoke",
                                "question": user_input,
                                "error": str(e)
                            })
                            st.error(f"오류 발생: 잠시 후 다시 시도해주세요.")
    else:
        st.info("👈 좌측에서 먼저 PDF 문서를 업로드해주세요.")