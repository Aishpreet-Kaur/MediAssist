"""
llm_chain.py
------------
Handles the LLM prompt construction and response generation
using Groq (free, fast LLaMA3) via LangChain.
Supports both single-turn diagnosis and multi-turn chat with memory.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


# ── Model Setup ────────────────────────────────────────────────────────────────
def get_llm():
    """Initialize and return the Groq LLM (LLaMA3 70B)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )
    return ChatGroq(
        model="llama3-70b-8192",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=1500,
    )


# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MediAssist, a compassionate and knowledgeable AI medical assistant.
Your role is to help people understand their symptoms and guide them toward appropriate care.

IMPORTANT RULES:
- You are NOT a replacement for professional medical advice.
- Always include a short disclaimer reminding users to consult a doctor for serious issues.
- Base your response ONLY on the provided medical context and general medical knowledge.
- Be empathetic, clear, and easy to understand. Avoid overly technical jargon.
- You have memory of the current conversation. Use prior context to give better, connected answers.
- If the user is following up on a previous symptom, acknowledge it and build upon the prior diagnosis.

FOR INITIAL SYMPTOM QUERIES, always structure your response using these sections:
1. POSSIBLE CONDITION(S)
2. HOME REMEDIES (safe things to try right now)
3. OTC MEDICINES (over-the-counter, no prescription needed)
4. SEE A DOCTOR IF... (warning signs that need professional attention)
5. DISCLAIMER

FOR FOLLOW-UP QUESTIONS (clarifications, what if, additional symptoms),
respond conversationally but clearly. You do not need to repeat the full structured format.
"""


# ── Single-turn Diagnosis (streaming) ─────────────────────────────────────────
def get_diagnosis_stream(symptoms: str, context: str, additional_info: str = "None provided"):
    """
    Streaming single-turn diagnosis. Used for the very first message.
    Yields text chunks for Streamlit real-time rendering.
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", """
A patient is experiencing the following symptoms:
{symptoms}

Patient additional info: {additional_info}

Relevant medical knowledge retrieved from knowledge base:
----- MEDICAL CONTEXT START -----
{context}
----- MEDICAL CONTEXT END -----

Please provide a structured response using the format described in your instructions.
""")
    ])

    chain = prompt | llm | StrOutputParser()
    for chunk in chain.stream({
        "symptoms": symptoms,
        "context": context,
        "additional_info": additional_info,
    }):
        yield chunk


# ── Multi-turn Chat with History (streaming) ──────────────────────────────────
def chat_with_history_stream(
    user_message: str,
    context: str,
    chat_history: list,
    additional_info: str = "None provided"
):
    """
    Multi-turn streaming chat with full conversation history.

    Args:
        user_message:   Current user message.
        context:        RAG-retrieved medical context for this message.
        chat_history:   List of dicts [{"role": "user"|"assistant", "content": "..."}]
        additional_info: Patient age/gender/conditions.

    Yields:
        Text chunks from the LLM.
    """
    llm = get_llm()

    # Build message list with system prompt at top
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Inject RAG context alongside the current user message
    augmented_message = f"""Patient info: {additional_info}

User message: {user_message}

Relevant medical knowledge for this query:
----- MEDICAL CONTEXT START -----
{context}
----- MEDICAL CONTEXT END -----
"""
    messages.append(HumanMessage(content=augmented_message))

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


if __name__ == "__main__":
    test_symptoms = "I have a high fever, body aches, and a runny nose since yesterday."
    test_context = "Fever: body temperature above 38C. Home remedies include hydration and rest."
    for chunk in get_diagnosis_stream(test_symptoms, test_context):
        print(chunk, end="", flush=True)
