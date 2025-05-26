# Streamlit PDPL Expert Chatbot (ChatGPT-style UI + Chat History + Custom Prompt Fix)

import os
import re
import json
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# === Setup ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="PDPL Legal Assistant", layout="wide")
st.title("🛡️ PDPL Legal Assistant Chatbot")

# === Extract PDPL Articles Function ===
def extract_articles(file_path: str, source_name: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])
    full_text += "\nArticle 9999 END"
    matches = re.findall(r"(Article\s+\d+)(.*?)(?=Article\s+\d+)", full_text, flags=re.DOTALL | re.IGNORECASE)

    documents = []
    for title, content in matches:
        article_number = title.split()[1]
        full_article = f"{title}{content}".strip()
        doc = Document(
            page_content=full_article,
            metadata={"source": source_name, "article": f"Article {article_number}"}
        )
        documents.append(doc)

    return documents

# === Logging to JSONL ===
def log_to_json(question, answer, sources, path="./logs.json"):
    log_entry = {
        "question": question,
        "answer": answer,
        "sources": [
            {"source": doc.metadata.get("source"), "article": doc.metadata.get("article")} 
            for doc in sources
        ]
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

# === Initialization Function ===
@st.cache_resource
def initialize():
    # Load documents
    doc_dir = "./documents"
    all_docs = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".pdf"):
            full_path = os.path.join(doc_dir, filename)
            source_name = os.path.splitext(filename)[0]
            articles = extract_articles(full_path, source_name)
            all_docs.extend(articles)

    # Embeddings + FAISS
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(all_docs, embedding_model)
    retriever = vector_store.as_retriever()

    # LLM and memory
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # Custom prompt
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
You are a banking legal advisor specialized in Saudi Arabia's PDPL.
Answer the user's question strictly based on the context provided from the law.
Cite the article number in your response. If unsure, escalate.
If the user ask you to provide an example, provide an example in a banking sector.
Here is the conversation so far:
{chat_history}

Context:
{context}

Question: {question}
Answer:
"""
    )

    # Conversational QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        output_key="answer"
    )

    return qa_chain, memory

qa_chain, memory = initialize()

# === Chat Interface ===
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

# Clear Chat Button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_display = []
    memory.clear()
    st.rerun()

# Display chat messages
for user_msg, bot_msg, sources in st.session_state.chat_display:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
        with st.expander("📚 Source Articles"):
            for doc in sources:
                st.markdown(f"- **{doc.metadata.get('source', 'Unknown')}** | *{doc.metadata.get('article', '')}*")

# Chat input
query = st.chat_input("Ask about the PDPL...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        result = qa_chain({"question": query})
        answer = result["answer"]
        sources = result.get("source_documents", [])

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("📚 Source Articles"):
            for doc in sources:
                st.markdown(f"- **{doc.metadata.get('source', 'Unknown')}** | *{doc.metadata.get('article', '')}*")

    # Save to display history only
    st.session_state.chat_display.append((query, answer, sources))

    # Log interaction to JSONL
    log_to_json(query, answer, sources)
