"""Step 5: Streamlit interface.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src import config, llm
from src.rag import answer
from src.retriever import Retriever
from src.skills import skill_gap

st.set_page_config(page_title="JobFit RAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading the job index...")
def get_retriever() -> Retriever:
    return Retriever()


def read_cv(upload) -> str:
    if upload is None:
        return config.CV_PATH.read_text(encoding="utf-8")
    if upload.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(upload).pages)
    return upload.read().decode("utf-8", errors="ignore")


retriever = get_retriever()
jobs = pd.read_csv(config.JOBS_CSV)

with st.sidebar:
    st.header("Settings")
    k = st.slider("Passages to retrieve (k)", 1, 10, config.TOP_K)
    company = st.selectbox("Filter by company", ["All companies"] + retriever.companies())
    st.caption(f"Embeddings: **{retriever.embedder.name}**  \nLLM: **{llm.describe()}**")
    if not llm.is_configured():
        st.info("Add an LLM API key in `.env` for written answers. Without one, the app shows the best passages.")

st.title("JobFit RAG")
st.write("Ask questions about saved job descriptions, or check your CV against a role.")

ask_tab, gap_tab, data_tab = st.tabs(["Ask a question", "Skill-gap check", "Jobs in the index"])

with ask_tab:
    examples = ["Which roles ask for Power BI?", "Which jobs need security clearance?",
                "Which internships want experience with LLMs or AI agents?",
                "What does AkzoNobel need for the Data Science placement?"]
    question = st.text_input("Your question", placeholder=examples[0])
    st.caption("Try: " + " · ".join(examples))
    if question:
        with st.spinner("Searching..."):
            result = answer(question, retriever, k=k,
                            company=None if company == "All companies" else company)
        if result["notice"]:
            st.warning(result["notice"])
        st.markdown(result["answer"])
        _, main_model = llm._settings()
        if result["model"] and result["model"] != main_model:
            st.caption(f"Answered by backup model {result['model']} (the main model is over its usage limit).")
        with st.expander(f"Sources ({len(result['sources'])} passages)"):
            for s in result["sources"]:
                st.markdown(f"**{s['company']} – {s['title']}** · score {s['score']} · closes {s['closing_date']}")
                st.caption(s["text"])

with gap_tab:
    labels = {f"{r.company} – {r.title}": r.job_id for r in jobs.itertuples()}
    choice = st.selectbox("Choose a job", list(labels))
    upload = st.file_uploader("Upload your CV (PDF or TXT). Leave empty to use the sample CV.", type=["pdf", "txt"])
    if st.button("Check my fit", type="primary"):
        result = skill_gap(labels[choice], read_cv(upload))
        st.metric("Skill coverage", f"{result['coverage']}%")
        left, right = st.columns(2)
        left.subheader("Skills you have")
        left.write("\n".join(f"- {s}" for s in result["have"]) or "None found")
        right.subheader("Skills to work on")
        right.write("\n".join(f"- {s}" for s in result["missing"]) or "None, great match")
        with st.expander("Eligibility and other information"):
            st.write(result["job"]["text"].split("Eligibility and other information")[-1].strip())
        if result["advice"]:
            st.subheader("Advice")
            st.markdown(result["advice"])
        elif result.get("notice"):
            st.info(result["notice"])

with data_tab:
    st.dataframe(jobs.drop(columns=["file"]), width="stretch",hide_index=True)
