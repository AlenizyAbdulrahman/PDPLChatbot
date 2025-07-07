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

## 📋 How It Works

- **PDF Loader** – Loads and parses PDPL documents into chunks

- **Embedding Engine** – Converts text chunks into vector embeddings

- **Vector Store** – Stores and searches relevant legal chunks

- **RAG Pipeline** – Uses user query + relevant chunks to generate an answer

- **LLM Response** – Final answer is generated and returned through Streamlit

---

## ✅ Use Cases

- Legal and compliance teams verifying PDPL rules

- Internal staff training on data privacy regulations

- Developers embedding privacy-by-design practices

- General public inquiries on PDPL rights and obligations

---

## 👤 Author

**Abdulrahman Alenizy**  
Senior AI Engineer, Arab National Bank  
📧 [alenizyabdulrahman@outlook.com](mailto:alenizyabdulrahman@outlook.com)  
🔗 [Linkedin](https://www.linkedin.com/in/abdulrahman-alenizy-51150a220/)

---

## ⚠️ Disclaimer

This chatbot is an educational and assistive tool. It **does not constitute legal advice**.  
For official interpretation of PDPL regulations, consult the Saudi Data & Artificial Intelligence Authority (SDAIA) or a licensed legal advisor.

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
