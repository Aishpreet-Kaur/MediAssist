# 🩺 MediAssist — AI-Powered Symptom Checker & Home Remedy Guide

> An intelligent medical assistant built with **LLM + RAG** that analyses your symptoms and provides home remedies, OTC medicine suggestions, and doctor-referral alerts.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green?style=flat-square)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-LLaMA3%2070B-purple?style=flat-square)

---

## 📌 Project Overview

MediAssist combines two core AI components:

| Component | Technology | Role |
|-----------|------------|------|
| **RAG** | FAISS + HuggingFace Embeddings | Retrieves relevant medical knowledge from a local knowledge base based on user symptoms |
| **LLM** | Groq (LLaMA3 70B) via LangChain | Generates structured diagnosis, home remedies, OTC medicine suggestions and doctor-alert advice |

---

## 🏗️ Project Architecture

```
User Describes Symptoms
        │
        ▼
┌─────────────────────┐
│   Streamlit UI      │  ← app.py
│   (Frontend)        │
└────────┬────────────┘
         │  symptoms text
         ▼
┌─────────────────────┐
│   RAG Pipeline      │  ← rag_pipeline.py
│   FAISS Vector DB   │
│   HuggingFace Emb.  │
└────────┬────────────┘
         │  top-5 relevant chunks
         ▼
┌─────────────────────┐
│   LLM Chain         │  ← llm_chain.py
│   Groq LLaMA3 70B   │
│   LangChain Prompt  │
└────────┬────────────┘
         │
         ▼
  Structured Response:
  🩺 Possible Conditions
  🏠 Home Remedies
  💊 OTC Medicines
  🚨 See a Doctor If...
  ⚕️ Disclaimer
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/mediassist.git
cd mediassist
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Now open .env and add your Groq API key
# Get a FREE key at: https://console.groq.com
```

### 5. Build the knowledge base (one-time)
```bash
python ingest.py
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
mediassist/
│
├── app.py                        # Streamlit frontend & UI
├── rag_pipeline.py               # Document loading, chunking, embedding, retrieval
├── llm_chain.py                  # LLM prompt template & response generation
├── ingest.py                     # Script to build vector store from docs
│
├── data/
│   └── medical_docs/
│       ├── common_illnesses_part1.txt   # Fever, cold, cough, headache
│       ├── common_illnesses_part2.txt   # Stomach, acidity, skin rash
│       └── common_illnesses_part3.txt   # Throat, body pain, emergencies
│
├── vectorstore/                  # FAISS index (auto-generated, gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔑 Getting Your Free Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up with Google/GitHub (free)
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key and paste it in your `.env` file:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```

---

## 💡 Features

- ✅ **RAG-powered** — answers grounded in medical knowledge base, not just LLM hallucination
- ✅ **Streaming responses** — real-time text generation
- ✅ **Structured output** — always returns home remedies, medicines, and doctor alerts
- ✅ **Context transparency** — shows retrieved RAG chunks for inspectability
- ✅ **Example symptoms** — one-click demo inputs
- ✅ **Patient info** — optionally include age, gender, existing conditions
- ✅ **Emergency warnings** — flags serious symptoms requiring immediate care
- ✅ **India-specific** — medicine names and emergency numbers relevant for India

---

## ⚠️ Disclaimer

> This application is built for **educational and demonstration purposes only**.
> It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare professional for medical decisions.
> **Emergency (India): 112 | Ambulance: 108**

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Meta LLaMA3 70B via Groq API
- **RAG Framework**: LangChain
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Language**: Python 3.10+
