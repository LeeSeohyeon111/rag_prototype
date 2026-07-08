import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class TechGPTState(TypedDict):
    question: str       
    intent: str 
    intents: list         
    context: str         
    answer: str          

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

def intent_analyzer_node(state: TechGPTState) -> TechGPTState:
    """
    ① Intent Analyzer Node
    사용자 질문의 의도를 파악합니다.
    - search: 특정 정보를 찾는 질문
    - summary: 문서 전체 요약 요청
    - general: 일반 대화

    [적용 기법]
    - 역할 지정 기법: AI에게 분류 전문가 역할 부여
    - 형식 지정 기법: 출력 형식을 단어 하나로 고정
    - Few Shot 기법: 예시 3개 이상 제공
    """
    prompt = ChatPromptTemplate.from_template("""
    #역할
    당신은 사용자 질문의 의도를 정확히 분류하는 전문 분석가입니다.

    #명령문
    아래 질문을 분석하여 의도를 분류하세요.

    #제약조건
    - search / summary / general 중 해당하는 것을 모두 골라 쉼표로 구분하세요
    - 다른 말은 절대 하지 마세요
    - 해당하는 의도가 여러 개면 전부 출력하세요
 
    #예시 (Few Shot)
    Q: 이 문서 요약해줘 → summary
    Q: 핵심 내용 정리해줘 → summary
    Q: 한 줄로 설명해줘 → summary
    Q: 사업 목적이 뭐야? → search
    Q: 예산이 얼마야? → search
    Q: 담당자가 누구야? → search
    Q: 안녕 → general
    Q: 고마워 → general
    Q: 요약하고 사업 목적도 알려줘 → summary,search
    Q: 정리해주고 예산도 알려줘 → summary,search
    Q: 안녕, 이 문서 요약해줘 → general,summary

    #입력문
    질문: {question}

    #출력형식
    의도 (쉼표 구분):""")
    
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": state["question"]}).strip().lower()
 
    valid = {"search", "summary", "general"}
    intents = [i.strip() for i in raw.split(",") if i.strip() in valid]
 
    if not intents:
        intents = ["general"]
 
    intent = intents[0]
 
    return {"intents": intents, "intent": intent}
 

def retriever_node(state: TechGPTState) -> TechGPTState:
    """
    ② RAG Agent Node
    Vector DB에서 관련 문서를 검색합니다.
 
    [적용 기법]
    - HyDE: 질문을 LLM으로 확장 후 검색
      사용자 질문 → LLM이 관련 키워드 확장
      → 확장된 키워드로 검색 → 더 잘 찾음
    """
    # 질문 확장 프롬프트
    expand_prompt = ChatPromptTemplate.from_template("""
    #역할
    당신은 문서 검색 전문가입니다.
 
    #명령문
    아래 질문과 관련된 검색 키워드를
    다양하게 확장해서 한 문장으로 만드세요.
 
    #제약조건
    - 원래 질문의 의미를 유지하세요
    - 유사어, 관련 단어를 포함하세요
    - 한국어로 작성하세요
    - 문장 하나만 출력하세요
 
    #예시
    Q: "프로젝트 배경이 뭐야"
    A: "사업 배경 필요성 추진 이유 목적 동기"
 
    Q: "예산이 얼마야"
    A: "사업 예산 비용 금액 예산액 총액 VAT"
 
    #입력문
    질문: {question}
 
    확장 키워드:
    """)
 
    expand_chain = expand_prompt | llm | StrOutputParser()
    expanded_query = expand_chain.invoke({"question": state["question"]})
 
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    docs = retriever.invoke(expanded_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def summary_node(state: TechGPTState) -> TechGPTState:
    """
    ③ Summary Agent Node
    문서 전체를 요약합니다.

    [적용 기법]
    - 역할 지정 기법: KEA 전문 요약가 역할 부여
    - 형식 지정 기법: 출력 형식 구체적으로 지정
    - Chain of Thought: 단계별 사고 과정 유도
    """
    prompt = ChatPromptTemplate.from_template("""
    #역할
    당신은 KEA(한국전자정보통신산업진흥회)의
    기술 문서 요약 전문가입니다.

    #명령문
    아래 문서를 단계적으로 분석하여 핵심만 요약하세요.

    #제약조건
    - 문서에 없는 내용은 절대 추가하지 마세요
    - 반드시 한국어로 작성하세요

    #입력문
    [문서 내용]
    {context}

    #Chain of Thought (단계별 사고)
    Step 1. 이 문서의 핵심 목적이 무엇인지 파악하세요
    Step 2. 주요 내용 3~5개를 추출하세요
    Step 3. 중요도 순으로 정렬하세요
    Step 4. 아래 형식으로 출력하세요

    #출력형식
    ** 핵심 목적** 
    (한 줄 요약)

    ** 주요 내용**
    - (핵심 내용 1)
    - (핵심 내용 2)
    - (핵심 내용 3)

    **핵심 키워드**
    (키워드 3~5개)
                                            
    내부 사고 과정(Step 1~4)은 출력하지 마세요

    """)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": state["context"]})
    return {"answer": answer}

def answer_node(state: TechGPTState) -> TechGPTState:
    """
    ④ NLG Node
    검색 결과를 바탕으로 질문에 답변합니다.

    [적용 기법]
    - 역할 지정 기법: KEA 기술 전문가 역할 부여
    - 형식 지정 기법: 출력 구조 명확하게 지정
    - Chain of Thought: 근거 기반 단계적 답변 유도
    """
    prompt = ChatPromptTemplate.from_template("""
    #역할
    당신은 KEA(한국전자정보통신산업진흥회)의
    기술 정보 전문 AI 어시스턴트입니다.
    기업·연구자에게 정확하고 신뢰할 수 있는 정보를 제공합니다.

    #명령문
    아래 문서를 기반으로 사용자 질문에 단계적으로 답변하세요.

    #제약조건
    - 반드시 문서에 있는 내용만 답하세요
    - 문서에 없는 내용은 절대 추가하지 마세요
    - 추측이나 가정은 하지 마세요
    - 반드시 한국어로 답하세요

    #입력문
    [참고 문서]
    {context}

    [사용자 질문]
    {question}

    #Chain of Thought (단계별 사고)
    Step 1. 질문의 핵심이 무엇인지 파악하세요
    Step 2. 문서에서 관련 내용을 찾으세요
    Step 3. 근거를 바탕으로 답변을 작성하세요
    
    Step 1~3은 내부 사고 과정입니다.
    최종 답변만 아래 형식으로 출력하세요.

    #출력형식
    **답변**
    (핵심 답변을 2~3줄로 작성)
 
    **근거**
    - (근거 1: 문서에서 인용)
    - (근거 2: 문서에서 인용)
 
    문서에 없는 내용은 "문서에서 찾을 수 없습니다" 라고 답하세요
    내부 사고 과정(Step 1~3)은 출력하지 마세요
    """)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    return {"answer": answer}

def general_node(state: TechGPTState) -> TechGPTState:
    """
    ⑤ General Node
    일반 대화를 처리합니다.

    [적용 기법]
    - 역할 지정 기법: KEA AI 어시스턴트 역할 부여
    - 이어쓰기 기법: 문서 기반 질문으로 자연스럽게 유도
    """
    prompt = ChatPromptTemplate.from_template("""
    #역할
    당신은 KEA(한국전자정보통신산업진흥회)의
    친절한 AI 어시스턴트입니다.

    #명령문
    사용자의 말에 친절하게 응답하고,
    자연스럽게 기술 정보 질문으로 대화를 이어가세요.

    #제약조건
    - 반드시 한국어로 답하세요
    - 너무 길게 답하지 마세요 (3줄 이내)
    - 항상 문서 기반 질문을 유도하는 말로 마무리하세요

    #입력문
    사용자: {question}

    #이어쓰기 기법
    AI 어시스턴트: 안녕하세요! 저는 KEA 기술 정보 AI입니다.
    (위 문장에 이어서 자연스럽게 답변을 완성하세요)
    """)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"]})
    return {"answer": answer}

def route_by_intent(state: TechGPTState) -> str:
    """
    Intent 분류 결과에 따라 다음 노드를 결정합니다.
    - summary  → retriever → summary_node
    - search   → retriever → answer_node
    - general  → general_node
    """
    return state["intent"]

def route_after_retrieval(state: TechGPTState) -> str:
    """
    검색 후 intent에 따라 요약 or 답변 노드로 분기
    """
    if state["intent"] == "summary":
        return "summary"
    return "answer"

def build_graph():
    graph = StateGraph(TechGPTState)
    
    graph.add_node("intent_analyzer", intent_analyzer_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("summary_agent", summary_node)
    graph.add_node("answer_agent", answer_node)
    graph.add_node("general_agent", general_node)
    
    graph.set_entry_point("intent_analyzer")
    
    graph.add_conditional_edges(
        "intent_analyzer",
        route_by_intent,
        {
            "summary": "retriever",
            "search": "retriever",
            "general": "general_agent"
        }
    )
    
    graph.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {
            "summary": "summary_agent",
            "answer": "answer_agent"
        }
    )
    
    graph.add_edge("summary_agent", END)
    graph.add_edge("answer_agent", END)
    graph.add_edge("general_agent", END)
    
    return graph.compile()

def process_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )
    split_docs = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

st.set_page_config(page_title="Tech-GPT", layout="centered")
st.title("Tech-GPT")
 
uploaded_file = st.file_uploader("PDF 업로드", type=["pdf"])
 
if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
 
    if "vectorstore" not in st.session_state:
        with st.spinner("문서 분석 중..."):
            st.session_state.vectorstore = process_pdf(temp_path)
            st.session_state.graph = build_graph()
        st.success("완료!")
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
 
    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
 
        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                result = st.session_state.graph.invoke({
                    "question": prompt,
                    "intent": "",
                    "intents": [],
                    "context": "",
                    "answer": ""
                })
 
                intents = result.get("intents", ["general"])
                final_answers = []
 
                for i in intents:
                    sub_result = st.session_state.graph.invoke({
                        "question": prompt,
                        "intent": i,
                        "intents": [i],
                        "context": result.get("context", ""),
                        "answer": ""
                    })
                    final_answers.append(sub_result["answer"])
 
                combined = "\n\n---\n\n".join(final_answers)
                st.markdown(combined)
 
        st.session_state.messages.append({
            "role": "assistant",
            "content": combined
        })
 
else:
    st.info("PDF를 업로드하면 대화가 시작됩니다.")