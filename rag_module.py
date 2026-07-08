import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# .env 파일에 저장된 API 키 로드
load_dotenv()

def create_rag_chain(pdf_path):
    # [1단계] 문서 로드 (Document Load)
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # [2단계] 문서 분할 (Text Split)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,         # 청크 사이즈 조절
        chunk_overlap=100,      # 오버랩 조절
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_documents = text_splitter.split_documents(docs)

    # [3~4단계] 임베딩 및 벡터 DB 저장 (Embedding & Vector DB)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    
    # [디버깅 메모] 아래 for문은 테스트 용도이므로 실제 함수 흐름에서는 생략하거나 pass 처리해야 함
    # for doc in vectorstore.similarity_search("투자"):
    #     print(doc.page_content) 

    # [5단계] 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # [6~7단계] 프롬프트 및 LLM 설정 (Prompt & LLM)
    template = """당신은 KEA(한국전자정보통신산업진흥회)의 
기술 정보 전문 AI 어시스턴트입니다.

아래 문서 내용을 바탕으로만 답변하세요.
문서에 없는 내용은 절대 답하지 마세요.

[참고 문서]
{context}

[사용자 질문]
{question}

[답변 규칙]
1. 핵심 내용을 먼저 답하세요
2. 문서에서 근거 문장을 인용하세요
3. 문서에 없으면 "문서에서 찾을 수 없습니다" 라고 답하세요
4. 한국어로 답하세요
5. 친절하고 전문적인 어투를 사용하세요"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # [8단계] 체인 생성 (Chain)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain