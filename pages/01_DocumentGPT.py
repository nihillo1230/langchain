from gc import callbacks
import time
from tkinter.messagebox import QUESTION
from typing import Dict, List
from uuid import UUID
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="😘",
    layout="centered",
)

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_path = f"./.cache/files/{file.name}"
    file_content = file.read()
    # st.write(file_content, file_path)

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=300,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_doc(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Answer the question using ONLY the following context. If you don't know the answer just say that you don't know. DON'T make anything up.

        Context : {context}
        """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocummentGPT")

st.markdown(
    """
    Welcome!

    Use this chatbot to ask question to an AI about your files!

    Upload your files on the sidebar.
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or docx file", type=["pdf", "txt", "docx", "csv"]
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything!!!")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_doc),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)


else:
    st.session_state["messages"] = []
