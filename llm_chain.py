"""
------------
Handles the LLM prompt construction and response generation
using Groq (free, fast LLaMA3) via LangChain.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Model Setup ────────────────────────────────────────────────────────────────
def get_llm():
    """Initialize and return the Groq LLM (LLaMA3 70B — free tier)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )
    return ChatGroq(
        model="llama3-70b-8192",
        groq_api_key=api_key,
        temperature=0.3,          # slightly creative but mostly factual
        max_tokens=1500,
    )


# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MediAssist, a compassionate and knowledgeable AI medical assistant.
Your role is to help people understand their symptoms and guide them toward appropriate care.

IMPORTANT RULES:
- You are NOT a replacement for professional medical advice.
- Always include a disclaimer reminding users to consult a doctor for serious issues.
- Base your response ONLY on the provided medical context and general medical knowledge.
- Be empathetic, clear, and easy to understand. Avoid overly technical jargon.
- Always structure your response in the exact format requested.

RESPONSE FORMAT (always use these exact sections):
1. 🩺 POSSIBLE CONDITION(S)
2. 🏠 HOME REMEDIES (safe things to try right now)
3. 💊 OTC MEDICINES (over-the-counter, no prescription needed)
4. 🚨 SEE A DOCTOR IF... (warning signs that need professional attention)
5. ⚕️ DISCLAIMER
"""

# ── Prompt Template ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """
A patient is experiencing the following symptoms:
{symptoms}

Patient additional info: {additional_info}

Here is relevant medical knowledge to assist your diagnosis:
----- MEDICAL CONTEXT START -----
{context}
----- MEDICAL CONTEXT END -----

Please provide a structured response following the format described in your instructions.
Be thorough but concise. Focus on practical, actionable advice.
""")
])


# ── Main Diagnosis Function ────────────────────────────────────────────────────
def get_diagnosis(symptoms: str, context: str, additional_info: str = "None provided") -> str:
    """
    Generate a medical diagnosis response using the LLM.

    Args:
        symptoms:        User-described symptoms string.
        context:         Retrieved RAG context from vector store.
        additional_info: Optional extra info (age, existing conditions, etc.)

    Returns:
        Formatted diagnosis string from the LLM.
    """
    llm = get_llm()
    chain = PROMPT_TEMPLATE | llm | StrOutputParser()
    response = chain.invoke({
        "symptoms": symptoms,
        "context": context,
        "additional_info": additional_info,
    })
    return response


# ── Streaming Version ──────────────────────────────────────────────────────────
def get_diagnosis_stream(symptoms: str, context: str, additional_info: str = "None provided"):
    """
    Streaming version of get_diagnosis for real-time Streamlit output.
    Yields text chunks as they arrive from the LLM.
    """
    llm = get_llm()
    chain = PROMPT_TEMPLATE | llm | StrOutputParser()
    for chunk in chain.stream({
        "symptoms": symptoms,
        "context": context,
        "additional_info": additional_info,
    }):
        yield chunk


if __name__ == "__main__":
    # Quick test
    test_symptoms = "I have a high fever, body aches, and a runny nose since yesterday."
    test_context = "Fever: body temperature above 38C. Home remedies include hydration and rest."
    print(get_diagnosis(test_symptoms, test_context))