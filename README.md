# 🌐 SmartTranslator

A small language-detection + translation app built with **LangChain** and **Mistral AI**.
Paste text in any language and it will (1) detect the language and (2) translate it to English — both at the same time.

It ships with two front-ends:

| File | Interface | Run with |
|------|-----------|----------|
| [main.py](main.py) | Command-line (REPL loop) | `python main.py` |
| [app.py](app.py) | Streamlit web UI | `streamlit run app.py` |

---

## How it works

Both entry points share the same core idea — two independent LangChain chains that run **in parallel** against the same input:

```
                    ┌─ detect_prompt   → model → parser ─┐
input text ──┬──────┤                                    ├──→ { language, translated }
             │      └─ translation_prompt → model → parser ┘
             │
        RunnableParallel
```

1. **Model** — `ChatMistralAI(model_name="mistral-small-2506")` is the LLM used for both tasks.
2. **Detection chain** — `detect_prompt | model | parser`. A tightly-scoped system prompt tells the model to reply with *only* the language name (e.g. `Hindi`, `French`), no explanation.
3. **Translation chain** — `translation_prompt | model | parser`. A second system prompt tells the model to return *only* the raw English translation, no commentary.
4. **`RunnableParallel`** — runs both chains concurrently and returns a dict:
   ```python
   { "language": "Hindi", "translated": "Hello, how are you?" }
   ```
5. **`StrOutputParser`** — strips the LangChain message wrapper so each chain returns a plain string.

### CLI (`main.py`)
Runs an infinite `input()` loop. Type text, press Enter, and it prints the detected language and translation. Enter `0` to quit.

### Web UI (`app.py`)
A styled Streamlit page:
- Text area for input + a **Translate** button.
- Shows the detected language as a badge and the translation in a card.
- Keeps an in-session **history** of past translations (cleared on refresh or via the **Clear** button).
- The chain is built once and cached with `@st.cache_resource`.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Mistral API key
Create a `.env` file in the project root:
```
MISTRAL_API_KEY="your-mistral-api-key-here"
```
`load_dotenv()` reads this at startup, and `langchain-mistralai` picks up `MISTRAL_API_KEY` automatically.

Get a key from the [Mistral console](https://console.mistral.ai/).

### 3. Run

**CLI:**
```bash
python main.py
```
```
You: Bonjour, comment ça va ?
Language: French
Translator: Hello, how are you?
You: 0   # quit
```

**Web app:**
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

---

## Project structure
```
smarttranslator/
├── main.py           # CLI version
├── app.py            # Streamlit web version
├── requirements.txt  # dependencies
├── .env              # MISTRAL_API_KEY (not committed)
└── README.md
```

---

## ⚠️ Security note
Never commit your real API key. Make sure `.env` is listed in `.gitignore`
(rename the existing `gitignore` file to `.gitignore`), and rotate the key if it has ever been pushed.
