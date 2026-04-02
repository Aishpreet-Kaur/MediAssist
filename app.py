"""
app.py
------
MediAssist — AI-Powered Symptom Checker & Home Remedy Guide
Main Streamlit application entry point.
"""

import streamlit as st
from rag_pipeline import load_vectorstore, retrieve_context
from llm_chain import get_diagnosis_stream

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

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    min-height: 100vh;
}

/* Header */
.mediassist-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}

.mediassist-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #e0f2fe;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}

.mediassist-header p {
    color: #7dd3fc;
    font-size: 1.1rem;
    font-weight: 300;
}

/* Cards */
.info-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Result box */
.result-box {
    background: rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    color: #e0f2fe;
    line-height: 1.8;
    font-size: 1rem;
}

/* Warning box */
.warning-box {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #fca5a5;
    font-size: 0.9rem;
    margin-top: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 32, 39, 0.95) !important;
    border-right: 1px solid rgba(125, 211, 252, 0.15);
}

section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100%;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5) !important;
}

/* Text areas & inputs */
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(125, 211, 252, 0.2) !important;
    border-radius: 10px !important;
    color: #e0f2fe !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Labels */
label {
    color: #7dd3fc !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Divider */
hr {
    border-color: rgba(125, 211, 252, 0.15) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #0ea5e9 !important;
}

/* Section headers in results */
.result-box h3 {
    color: #7dd3fc;
    font-family: 'DM Serif Display', serif;
}
</style>
""", unsafe_allow_html=True)


# ── Load Vector Store (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading medical knowledge base...")
def load_vs():
    return load_vectorstore()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 MediAssist")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. **Describe** your symptoms below
2. **RAG** retrieves relevant medical knowledge
3. **AI** generates personalized advice
4. **Act** on home remedies or see a doctor
    """)
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
This tool is for **educational purposes only**.  
It is **NOT** a substitute for professional medical advice.  

For **emergencies**, call **112** (India).
    """)
    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    st.markdown("""
- Fever & Common Cold
- Cough & Sore Throat
- Headache & Body Pain
- Stomach Issues & Acidity
- Skin Rash & Allergies
- Emergency Warning Signs
    """)


# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mediassist-header">
    <h1>🩺 MediAssist</h1>
    <p>AI-powered symptom checker with RAG — get home remedies, medicine suggestions & doctor alerts</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Input Form ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 📝 Describe Your Symptoms")
    symptoms = st.text_area(
        label="symptoms_input",
        placeholder="e.g. I have a high fever since yesterday, my whole body is aching, I have a runny nose and I feel very tired...",
        height=160,
        label_visibility="collapsed",
    )

with col2:
    st.markdown("#### 👤 Patient Info (optional)")
    age = st.text_input("Age", placeholder="e.g. 25")
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    existing = st.text_input("Existing conditions", placeholder="e.g. Diabetes, Asthma")

    additional_info = f"Age: {age or 'Not provided'}, Gender: {gender}, Existing conditions: {existing or 'None'}"

st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    diagnose_btn = st.button("🔍 Analyse Symptoms", use_container_width=True)

# ── Example Symptom Chips ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Try an example:**")
ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)

examples = {
    "🤒 Fever + Cold": "I have a high fever of around 102°F, runny nose, sneezing and body aches since 2 days.",
    "🤢 Stomach Ache": "I have severe stomach ache, bloating, nausea and loose stools since this morning.",
    "🤕 Headache": "I have a throbbing headache on one side, sensitivity to light and mild nausea.",
    "😮‍💨 Cough + Throat": "I have a dry cough, sore throat and slight fever for the past 3 days.",
}

with ex_col1:
    if st.button(list(examples.keys())[0], use_container_width=True):
        st.session_state["example_symptoms"] = list(examples.values())[0]
with ex_col2:
    if st.button(list(examples.keys())[1], use_container_width=True):
        st.session_state["example_symptoms"] = list(examples.values())[1]
with ex_col3:
    if st.button(list(examples.keys())[2], use_container_width=True):
        st.session_state["example_symptoms"] = list(examples.values())[2]
with ex_col4:
    if st.button(list(examples.keys())[3], use_container_width=True):
        st.session_state["example_symptoms"] = list(examples.values())[3]

# Auto-fill example into text area
if "example_symptoms" in st.session_state:
    symptoms = st.session_state["example_symptoms"]
    st.info(f"💡 Example loaded: _{symptoms}_")

# ── Diagnosis Output ───────────────────────────────────────────────────────────
if diagnose_btn or ("example_symptoms" in st.session_state and symptoms):
    if not symptoms.strip():
        st.warning("⚠️ Please describe your symptoms before analysing.")
    else:
        st.markdown("---")
        st.markdown("#### 🧠 AI Diagnosis")

        # Load vector store
        try:
            vectorstore = load_vs()
        except Exception as e:
            st.error(f"❌ Failed to load knowledge base: {e}\n\nRun `python ingest.py` first!")
            st.stop()

        # Retrieve context via RAG
        with st.spinner("🔍 Searching medical knowledge base..."):
            context = retrieve_context(symptoms, vectorstore, k=5)

        # Show retrieved context in expander (transparency)
        with st.expander("📚 Retrieved Medical Context (RAG)", expanded=False):
            st.markdown(f"```\n{context[:1500]}...\n```")

        # Stream LLM response
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        try:
            result_placeholder = st.empty()
            full_response = ""
            for chunk in get_diagnosis_stream(symptoms, context, additional_info):
                full_response += chunk
                result_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"❌ LLM Error: {e}")
            st.markdown("Make sure your `GROQ_API_KEY` is set correctly in `.env`")
        st.markdown('</div>', unsafe_allow_html=True)

        # Emergency warning
        st.markdown("""
<div class="warning-box">
⚠️ <strong>Important:</strong> This is AI-generated information for educational purposes only.
If symptoms are severe, worsening, or you are unsure — please consult a qualified doctor.
<strong>Emergency (India): 112 | Ambulance: 108</strong>
</div>
""", unsafe_allow_html=True)

        # Clear example after use
        if "example_symptoms" in st.session_state:
            del st.session_state["example_symptoms"]

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color: #475569; font-size:0.85rem; padding-bottom:1rem;'>
    Built with ❤️ using LangChain · FAISS · Groq LLaMA3 · Streamlit &nbsp;|&nbsp;
    For educational use only
</div>
""", unsafe_allow_html=True)