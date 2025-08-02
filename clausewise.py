import streamlit as st
from transformers import pipeline
import docx2txt
import PyPDF2
import tempfile
import base64

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="ClauseWise", page_icon="📜", layout="wide")

# -------------------- UTILS --------------------
def get_base64(file_bytes):
    return base64.b64encode(file_bytes).decode()

def get_logo_base64(uploaded_logo):
    if uploaded_logo:
        return get_base64(uploaded_logo.read())
    else:
        with open("logo.png", "rb") as f:
            return get_base64(f.read())

# -------------------- SIDEBAR UPLOAD --------------------
st.sidebar.markdown("### ⚙️ Customize Logo")
uploaded_logo = st.sidebar.file_uploader("Upload a Logo Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
logo_base64 = get_logo_base64(uploaded_logo)

# -------------------- CSS --------------------
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&display=swap" rel="stylesheet">
<style>
html, body, .stApp {{
    background-color: #1a1a1a !important;
    font-family: 'Helvetica Neue', sans-serif;
    color: #ffffff;
}}
#MainMenu, footer {{visibility: hidden;}}
.stTextInput > div > div > input,
.stTextArea > div > textarea {{
    background-color: #444654 !important;
    color: #fff;
    border-radius: 8px;
    padding: 12px;
    border: none;
    margin-bottom: 1rem;
    font-size: 16px;
}}
.stButton>button {{
    background-color: #000000;
    color: white;
    border: 1px solid #555;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 500;
    cursor: pointer;
    box-shadow: 0 0 6px #111;
    transition: all 0.2s ease-in-out;
}}
.stButton>button:hover {{
    background-color: #111111;
    border-color: #888;
    transform: scale(1.02);
    box-shadow: 0 0 10px #333;
}}
.header {{
    background-color: #000000;
    padding: 1.5rem 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.header-left {{
    display: flex;
    align-items: center;
}}
.header h1 {{
    font-family: 'Space+Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    color: white;
    margin: 0;
}}
.header h1 .main-title {{
    font-size: 48px;
}}
.trigon-text {{
    font-family: 'Space+Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: #00BFFF;
    margin-left: auto;
}}
.logo-img {{
    height: 80px;
    width: 80px;
    margin-right: 1rem;
    border-radius: 8px;
    object-fit: cover;
}}
</style>
<div class="header">
    <div class="header-left">
        <img class="logo-img" src="data:image/png;base64,{logo_base64}" alt="Logo">
        <h1><span class="main-title">ClauseWise</span>: AI Legal Document Analyzer</h1>
    </div>
    <div class="trigon-text">by TRIGON</div>
</div>
""", unsafe_allow_html=True)

# -------------------- ABOUT --------------------
st.subheader("📘 About This App")
st.markdown("""
**ClauseWise is your intelligent legal assistant that simplifies the complex.**  
It analyzes legal documents—like contracts, NDAs, leases, and agreements—using AI to break down dense clauses, highlight key entities, simplify legal jargon, and classify the type of document instantly.  
With an intuitive interface, you can upload files, ask questions, and receive structured insights in seconds—making legal review faster, clearer, and smarter.
""")

# -------------------- CHAT SETUP --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Accurate legal QA model
legal_advice_generator = pipeline("question-answering", model="deepset/roberta-base-squad2")
simplifier = pipeline("text2text-generation", model="t5-small")
ner_model = pipeline("ner", grouped_entities=True)
classifier = pipeline("zero-shot-classification")

# -------------------- DOCUMENT ANALYSIS (Moved Up) --------------------
with st.form("query_form"):
    st.markdown("""
    <div style="background-color: #1c1e24; padding: 20px; border-radius: 12px; margin-top: 30px;">
        <h4 style="color: white;">📄 Upload your document and ask a question</h4>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["pdf", "docx", "txt"])
    user_query = st.text_area("Ask your question here:", height=150, placeholder="E.g. What clauses limit liability?")
    doc_submit = st.form_submit_button("Submit")

def read_file(file):
    if file.name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        with open(tmp_file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "Unsupported file format."

if doc_submit and uploaded_file:
    try:
        document_text = read_file(uploaded_file)

        if user_query:
            st.subheader("🔍 Your Query")
            st.write(f"You asked: *{user_query}*")
            answer = legal_advice_generator(question=user_query, context=document_text)
            st.subheader("💬 AI's Answer")
            st.success(answer["answer"])

        st.subheader("🗞 Clause Breakdown")
        clauses = [c.strip() for c in document_text.split("\n") if len(c.strip()) > 50]
        for i, clause in enumerate(clauses[:5]):
            st.markdown(f"**Clause {i+1}:** {clause}")

        st.subheader("💡 Simplified Clauses")
        for clause in clauses[:3]:
            result = simplifier(clause, max_length=100, do_sample=False)[0]["generated_text"]
            st.write(result)

        st.subheader("🏷️ Named Entities")
        entities = ner_model(document_text)
        for ent in entities:
            st.write(f"{ent['entity_group']}: {ent['word']} ({ent['score']:.2f})")

        st.subheader("📁 Document Type Classification")
        candidate_labels = ["NDA", "lease", "employment contract", "service agreement"]
        classification = classifier(document_text, candidate_labels=candidate_labels)
        st.json(classification)

    except Exception as e:
        st.error(f"An error occurred: {e}")
elif doc_submit and not uploaded_file:
    st.warning("Please upload a legal document to proceed.")

# -------------------- LEGAL Q&A (Moved Below) --------------------
st.markdown("### 🧠 Ask Legal Questions (AI-Powered Chat)")

with st.form("chat_form"):
    user_input = st.text_area("Ask a legal question:", height=120, placeholder="E.g. Is a verbal agreement legally binding?")
    submitted = st.form_submit_button("Send")

default_context = (
    "In most legal systems, a contract is an agreement between parties that is legally binding. "
    "A verbal contract can be legally binding, but it may be harder to prove in court compared to a written agreement. "
    "Certain types of agreements, such as real estate transactions, may require a written contract under the Statute of Frauds."
)

if submitted and user_input.strip():
    question = user_input.strip()
    st.session_state.chat_history.append(("user", question))
    try:
        ai_raw = legal_advice_generator(question=question, context=default_context)
        answer = ai_raw["answer"]
    except:
        answer = "Sorry, I couldn't generate a reliable answer right now."
    st.session_state.chat_history.append(("ai", answer.strip()))

for speaker, message in st.session_state.chat_history:
    if speaker == "user":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 ClauseWise AI:** {message}")

if st.session_state.chat_history:
    st.markdown("""
    <div style="color: orange; background-color: #1f1f1f; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
    ⚠️ <strong>Disclaimer:</strong> The information provided by ClauseWise is AI-generated and is for informational purposes only. It does not constitute legal advice. Always consult a licensed attorney.
    </div>
    """, unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("<p style='text-align: center; color: grey;'>© 2025 ClauseWise AI</p>", unsafe_allow_html=True)
