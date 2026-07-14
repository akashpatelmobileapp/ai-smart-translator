from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="LinguaAI — Translate Anything",
    page_icon="🌐",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e8e8f0;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* App container */
.block-container {
    max-width: 720px;
    padding: 2rem 1.5rem 4rem;
}

/* Hero title */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Input area */
.stTextArea textarea {
    background: #1a1d2e !important;
    border: 1.5px solid #2d2f45 !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 1rem !important;
    resize: none !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}

/* Translate button */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* Result cards */
.result-card {
    background: #1a1d2e;
    border: 1.5px solid #2d2f45;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}

.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

.lang-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed22, #3b82f622);
    border: 1px solid #7c3aed55;
    color: #a78bfa;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.3rem 1rem;
    border-radius: 8px;
    letter-spacing: 0.02em;
}

.translated-text {
    font-size: 1.05rem;
    color: #e8e8f0;
    line-height: 1.7;
    font-weight: 400;
}

/* History section */
.history-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4b5563;
    margin: 2.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.history-item {
    background: #13151f;
    border: 1px solid #1f2235;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
}

.history-input {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-meta {
    font-size: 0.8rem;
    color: #a78bfa;
    font-weight: 500;
}

.divider {
    border: none;
    border-top: 1px solid #1f2235;
    margin: 2rem 0;
}

/* Clear button */
.stButton.clear > button {
    background: transparent !important;
    border: 1px solid #2d2f45 !important;
    color: #6b7280 !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 1rem !important;
    width: auto !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # temporary in-session memory

# ── LangChain setup ───────────────────────────────────────
@st.cache_resource
def get_chain():
    model  = ChatMistralAI(model_name="mistral-small-2506")
    parser = StrOutputParser()

    detect_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a language detection tool.
Your ONLY job is to identify the language of the given text.
Rules:
- Reply with ONLY the language name. Example: Hindi, Gujarati, French, Spanish
- Do NOT translate
- Do NOT explain
- Do NOT add punctuation or extra words
- Just one word: the language name"""),
        ("human", "{text}")
    ])

    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a translation tool.
Your ONLY job is to translate the given text to English.
Rules:
- Reply with ONLY the translated English text
- Do NOT explain the translation
- Do NOT add notes or comments
- Do NOT say "Here is the translation" or anything like that
- Just return the raw translated text and nothing else"""),
        ("human", "{text}")
    ])

    detect_chain    = detect_prompt    | model | parser
    translate_chain = translation_prompt | model | parser

    return RunnableParallel({
        "language"  : detect_chain,
        "translated": translate_chain,
    })

chain = get_chain()

# ── Hero ──────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌐 Language AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Paste any text — detect the language and translate it to English instantly.</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────
query = st.text_area(
    label="Input text",
    placeholder="Type or paste text in any language here...",
    height=140,
    label_visibility="collapsed",
)

translate_clicked = st.button("Translate →")

# ── Run chain ─────────────────────────────────────────────
if translate_clicked:
    if not query.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Detecting & translating..."):
            result = chain.invoke({"text": query})

        language   = result["language"]
        translated = result["translated"]

        # Save to session memory
        st.session_state.history.insert(0, {
            "input"     : query,
            "language"  : language,
            "translated": translated,
        })

        # ── Results ───────────────────────────────────────
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Detected Language</div>
                <div class="lang-badge">{language}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">English Translation</div>
                <div class="translated-text">{translated}</div>
            </div>
            """, unsafe_allow_html=True)

# ── History ───────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown('<div class="history-header">⏱ This session</div>', unsafe_allow_html=True)
    with h_col2:
        if st.button("Clear", key="clear"):
            st.session_state.history = []
            st.rerun()

    for item in st.session_state.history:
        preview = item["input"][:60] + ("..." if len(item["input"]) > 60 else "")
        st.markdown(f"""
        <div class="history-item">
            <div class="history-input">"{preview}"</div>
            <div class="history-meta">{item['language']} → {item['translated'][:80]}{'...' if len(item['translated']) > 80 else ''}</div>
        </div>
        """, unsafe_allow_html=True)