import globals
import os
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
import streamlit as st
import time
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

if "history" not in st.session_state:
    st.session_state.history = []


# INSERT YOUR HUGGING FACE TOKEN HERE
HUGGING_TOKEN = ""
globals.get_hf_token_if_empty(HUGGING_TOKEN)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
ENDPOINT_URL = f"https://api-inference.huggingface.co/models/{repo_id}"
llm = HuggingFaceEndpoint(
        endpoint_url=ENDPOINT_URL,
        temperature = 0.1,
        max_new_tokens=240,       
)
time.sleep(2)

persist_directory = "./db/mistral/"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


if not os.path.exists(persist_directory):
    with st.spinner("Wait for the database to be initialized..."):
        data = globals.load_pdf_file(
            file_path_pdf="app\PCR FIGT CRL - Campionato Regionale 2024-2025.pdf"
        )

        db_chroma = Chroma.from_documents(data, embeddings, persist_directory=persist_directory)
        db_chroma.persist()
        print("processing done")

elif os.path.exists(persist_directory):
    db_chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("db uploaded")




retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db_chroma.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
)


for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

prompt = st.chat_input("Ask something")

if prompt:
    st.session_state.history.append(
        {
            'role':'user',
            'content':prompt
        }
    )

    with st.chat_message('user'):
        st.markdown(prompt)
    
    with st.spinner("Thinking..."):
        response = retrieval_chain(
            {
                'query':prompt
            }
        )

        st.session_state.history.append(
            {
                'role':'assistant',
                'content':response['result']
            }
        )

        with st.chat_message('assistant'):
            st.markdown(response['result'])