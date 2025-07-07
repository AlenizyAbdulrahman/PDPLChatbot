# 🛡️ PDPL Chatbot – Your Saudi Data Privacy Assistant 🇸🇦

The **PDPL Chatbot** is an AI-powered assistant designed to help users understand and navigate the **Personal Data Protection Law (PDPL)** in Saudi Arabia. Built using Retrieval-Augmented Generation (RAG), the chatbot can answer queries related to PDPL regulations, definitions, data subject rights, data controller obligations, cross-border data transfers, penalties, and more.

---

## 🚀 Live Demo

🔗 [Click here to interact with the PDPL Chatbot](https://pdplchatbot-lzajdavcdmruha9y3qvwy3.streamlit.app/)

---

## 🧠 Features

- 💬 Natural language Q&A based on official PDPL legal documents
- 🔍 Accurate retrieval using document chunking and embedding
- 🧾 Handles PDF regulations and extracts relevant legal context
- 🌐 Arabic & English support (if applicable)
- 🛡️ Ensures regulatory compliance and mitigates hallucinations with guardrails

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **LLM**: OpenAI 
- **Vector Store**: FAISS 
- **Embedding**: OpenAIEmbeddings
- **Document Source**: Official PDPL PDFs

---

## 📦 Installation (For Local Development)

```bash
# Clone the repository
git clone https://github.com/AlenizyAbdulrahman/PDPLChatbot.git
cd PDPLChatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run second-app.py
