import json
import os
import time
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

def get_hf_token_if_empty(HF_TOKEN):

    if HF_TOKEN != "":
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    
    else:
        with open("app\credentials.json","r") as file:
            f = json.load(file)
            HF_TOKEN = f["HUGGING_TOKEN"]
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN



def setup_llm(
       repo_id: str,
       temperature: float, 
):
    ENDPOINT_URL = f"https://api-inference.huggingface.co/models/{repo_id}"

    llm = HuggingFaceEndpoint(
        endpoint_url=ENDPOINT_URL,
        temperature = temperature,
        max_new_tokens = 249,
        inputs = 32768
        
    )
    time.sleep(2)

    return llm



def load_pdf_file(
        file_path_pdf: str   
):
    loader = PDFMinerLoader(file_path=file_path_pdf)
    data = loader.load()
    return data


def embed_document(
    data,
    persist_directory
):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
    return db, embedding_function



def create_rag_chain(
        db,
        llm
):
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain


def create_conversational_chain(
       db,
       llm, 
):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original English.
                            Chat History:
                            {chat_history}
                            Follow-Up Input: {question}
                            Standalone question:"""

    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        memory=memory,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT
    )

    return conversational_chain