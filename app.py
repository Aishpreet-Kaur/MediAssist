"""
app.py
------
MediAssist — AI-Powered Symptom Checker & Home Remedy Guide
Multi-turn chat interface with RAG + LLM + conversation memory.
"""

import streamlit as st
from rag_pipeline import load_vectorstore, retrieve_context
from llm_chain import get_diagnosis_stream, chat_with_history_stream

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediAssist — AI Symptom Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    min-height: 100vh;
}

.mediassist-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.mediassist-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #e0f2fe;
    margin-bottom: 0.2rem;
}
.mediassist-header p {
    color: #7dd3fc;
    font-size: 1rem;
    font-weight: 300;
}

/* Chat messages */
.chat-user {
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: #e0f2fe;
    font-size: 0.97rem;
    line-height: 1.6;
}
.chat-assistant {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px 16px 16px 16px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: #e0f2fe;
    font-size: 0.97rem;
    line-height: 1.8;
}
.chat-label-user {
    text-align: right;
    color: #7dd3fc;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    letter-spacing: 0.05em;
}
.chat-label-assistant {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    letter-spacing: 0.05em;
}

.warning-box {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    color: #fca5a5;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 32, 39, 0.97) !important;
    border-right: 1px solid rgba(125, 211, 252, 0.1);
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(14, 165, 233, 0.45) !important;
}

/* Input */
.stTextInput input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(125,211,252,0.25) !important;
    border-radius: 12px !important;
    color: #e0f2fe !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    padding: 0.8rem 1rem !important;
}
.stTextInput input::placeholder { color: #64748b !important; }

label { color: #7dd3fc !important; font-weight: 500 !important; }
hr { border-color: rgba(125, 211, 252, 0.12) !important; }

/* Empty chat area */
.chat-empty {
    text-align: center;
    padding: 3rem 2rem;
    color: #475569;
}
.chat-empty h3 { color: #64748b; font-family: 'DM Serif Display', serif; font-size: 1.4rem; }

/* RAG badge */
.rag-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
    border-radius: 20px;
    padding: 0.1rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

/* Scrollable chat container */
.chat-scroll {
    max-height: 62vh;
    overflow-y: auto;
    padding-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # [{"role": ..., "content": ...}]
if "patient_info" not in st.session_state:
    st.session_state.patient_info = "Not provided"
if "show_rag_ctx" not in st.session_state:
    st.session_state.show_rag_ctx = {}       # {msg_index: context_str}


# ── Load Vector Store (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading medical knowledge base...")
def load_vs():
    return load_vectorstore()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 MediAssist")
    st.markdown("---")

    st.markdown("### 👤 Patient Info")
    age = st.text_input("Age", placeholder="e.g. 25", key="age_input")
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    existing = st.text_input("Existing conditions", placeholder="e.g. Diabetes")

    if age or existing:
        st.session_state.patient_info = (
            f"Age: {age or 'Not provided'}, "
            f"Gender: {gender}, "
            f"Existing conditions: {existing or 'None'}"
        )

    st.markdown("---")
    st.markdown("### 💬 Chat Controls")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.show_rag_ctx = {}
        st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    st.markdown("""
Topics covered:
- Fever, Cold, Cough, Headache
- Stomach, Acidity, Diarrhea
- Skin Rash, Cuts & Wounds
- Dengue, Malaria, Typhoid, Chikungunya
- Diabetes, Blood Pressure, Anemia
- Mental Health, Anxiety, Sleep
- Women's Health, Pediatric
- Emergency Warning Signs
    """)
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("For **educational use only**. Not a substitute for professional medical advice. Emergency: **112**")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mediassist-header">
    <h1>🩺 MediAssist</h1>
    <p>Describe your symptoms and chat with an AI doctor — powered by RAG + LLaMA3</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Example chips ──────────────────────────────────────────────────────────────
examples = {
    "🤒 Fever + Body ache": "I have had a high fever of around 102F, body aches all over, and a runny nose since yesterday. What could it be?",
    "🦟 Dengue symptoms": "I have had sudden high fever for 2 days, severe headache, pain behind my eyes and my joints are very painful. Could it be dengue?",
    "😟 Anxiety & stress": "I have been feeling very anxious lately, my heart races for no reason, I cannot sleep and feel restless all the time.",
    "🩸 Low energy + pale": "I feel very tired all the time, I look pale, get breathless easily even on walking, and my nails look white. What is wrong?",
}

ex_cols = st.columns(4)
selected_example = None
for i, (label, text) in enumerate(examples.items()):
    with ex_cols[i]:
        if st.button(label, use_container_width=True, key=f"ex_{i}"):
            selected_example = text

st.markdown("---")

# ── Chat Display ───────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
<div class="chat-empty">
    <h3>How can I help you today?</h3>
    <p>Describe your symptoms below or tap an example above to get started.<br>
    You can follow up with more questions — I remember our conversation.</p>
</div>
""", unsafe_allow_html=True)
    else:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-label-user">YOU</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="chat-label-assistant">🩺 MEDIASSIST '
                    f'<span class="rag-badge">RAG</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

                # Show RAG context toggle for this message
                ctx_key = str(i)
                if ctx_key in st.session_state.show_rag_ctx:
                    with st.expander("📚 View retrieved medical context", expanded=False):
                        st.code(st.session_state.show_rag_ctx[ctx_key][:1200] + "...", language=None)

st.markdown("")

# ── Chat Input ─────────────────────────────────────────────────────────────────
col_input, col_send = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        label="message",
        placeholder="Describe your symptoms or ask a follow-up question...",
        label_visibility="collapsed",
        key="user_input_field",
        value=selected_example or "",
    )

with col_send:
    send_btn = st.button("Send 📤", use_container_width=True)

# ── Process Message ────────────────────────────────────────────────────────────
trigger = send_btn or (selected_example is not None)
message_to_send = user_input.strip() if not selected_example else selected_example.strip()

if trigger and message_to_send:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": message_to_send,
    })

    # Load vector store
    try:
        vectorstore = load_vs()
    except Exception as e:
        st.error(f"Knowledge base error: {e}. Run `python ingest.py` first.")
        st.stop()

    # RAG retrieval
    context = retrieve_context(message_to_send, vectorstore, k=5)

    # Stream LLM response
    is_first_message = len(st.session_state.chat_history) == 1

    with st.spinner("🔍 Analysing symptoms..."):
        full_response = ""
        response_placeholder = st.empty()

        try:
            if is_first_message:
                stream_fn = get_diagnosis_stream(
                    symptoms=message_to_send,
                    context=context,
                    additional_info=st.session_state.patient_info,
                )
            else:
                # Pass history EXCLUDING the just-added user message
                history_so_far = st.session_state.chat_history[:-1]
                stream_fn = chat_with_history_stream(
                    user_message=message_to_send,
                    context=context,
                    chat_history=history_so_far,
                    additional_info=st.session_state.patient_info,
                )

            for chunk in stream_fn:
                full_response += chunk
                response_placeholder.markdown(
                    f'<div class="chat-assistant">{full_response}▌</div>',
                    unsafe_allow_html=True,
                )

            response_placeholder.empty()

        except Exception as e:
            full_response = f"Sorry, I encountered an error: {e}\n\nPlease check your GROQ_API_KEY in the .env file."

    # Save assistant response + RAG context
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": full_response,
    })
    ctx_index = str(len(st.session_state.chat_history) - 1)
    st.session_state.show_rag_ctx[ctx_index] = context

    st.rerun()

# ── Emergency Footer ───────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("""
<div class="warning-box">
⚠️ <strong>Reminder:</strong> MediAssist is for educational purposes only — not a substitute for professional medical advice.
If symptoms are severe or worsening, please see a doctor. <strong>Emergency (India): 112 | Ambulance: 108 | Mental Health: iCall 9152987821</strong>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#475569; font-size:0.82rem; padding-bottom:1rem;'>
    Built with LangChain · FAISS · HuggingFace Embeddings · Groq LLaMA3 · Streamlit
    &nbsp;|&nbsp; For educational use only
</div>
""", unsafe_allow_html=True)
