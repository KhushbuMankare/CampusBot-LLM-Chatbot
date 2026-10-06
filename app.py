"""
app.py - Streamlit interface for the LLM Prompt Engineering Chatbot.

Run with:  streamlit run app.py
"""

import os
import time

import pandas as pd
import streamlit as st

from evaluation import run_evaluation, summarize_results
from llm import MODEL_NAME, api_key_available, ask_llm
from prompts import TASKS, TECHNIQUES, TECHNIQUE_INFO, build_prompt, faq_to_text

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

st.set_page_config(page_title="CampusBot - Prompt Engineering", page_icon="🎓", layout="wide")


@st.cache_data
def load_data():
    faq = pd.read_csv(os.path.join(DATA_DIR, "faq.csv"))
    tests = pd.read_csv(os.path.join(DATA_DIR, "test_queries.csv"))
    return faq, tests


faq_df, test_df = load_data()
faq_text = faq_to_text(faq_df)

# Text shown to the user for each task
HINTS = {
    "Q&A": "Ask a question about the college (fees, exams, library, hostel, placements...).",
    "Summarization": "Paste a paragraph and the bot will summarize it.",
    "Classification": "Type a student query and the bot will tell which category it belongs to.",
}
PLACEHOLDERS = {
    "Q&A": "e.g. What is the annual tuition fee?",
    "Summarization": "Paste the text you want to summarize...",
    "Classification": "e.g. My student portal password is not working",
}
SAMPLE_INPUTS = {
    "Q&A": "What attendance do I need to sit for the exams?",
    "Summarization": (
        "The college will organise a three-day technical fest next month. Students can take part in "
        "coding contests, robotics workshops and project exhibitions. Registration is free for first-year "
        "students, while other students must pay Rs. 200. The best projects will win cash prizes, and "
        "several companies will visit the fest to meet participants for internships."
    ),
    "Classification": "Can I get a hostel room if I take admission late?",
}

# ------------------------------------------------------------ Header
st.title("🎓 CampusBot: LLM Chatbot with Prompt Engineering")
st.caption("Q&A, summarization and classification using Gemini, with prompt comparison and evaluation.")

if not api_key_available():
    st.error("GEMINI_API_KEY not found. Create a `.env` file in the project folder and add your key. "
             "See the README for the steps, then restart the app.")
    st.stop()

# ----------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    task = st.selectbox("Task", TASKS)
    technique = st.selectbox("Prompt technique", TECHNIQUES, index=3)
    st.caption(TECHNIQUE_INFO[technique])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1,
                            help="Low = predictable answers, high = more varied answers.")
    st.caption(f"Model: `{MODEL_NAME}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_message(msg):
    """Display one chat message (and the prompt used, for assistant messages)."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.caption(f"Task: {msg['task']} | Technique: {msg['technique']}")
            with st.expander("Show prompt used"):
                if msg["system"]:
                    st.text("SYSTEM:\n" + msg["system"])
                st.text("PROMPT:\n" + msg["prompt"])


tab_chat, tab_compare, tab_eval, tab_data = st.tabs(
    ["💬 Chatbot", "🔍 Compare Prompts", "📊 Evaluation", "📁 Dataset"]
)

# ------------------------------------------------------- Tab 1: Chatbot
with tab_chat:
    st.subheader(f"{task} mode")
    st.caption(HINTS[task] + " Each message is answered on its own (no chat memory).")

    for msg in st.session_state.messages:
        show_message(msg)

    user_text = st.chat_input(PLACEHOLDERS[task])
    if user_text:
        user_msg = {"role": "user", "content": user_text}
        show_message(user_msg)
        st.session_state.messages.append(user_msg)

        system, prompt = build_prompt(task, technique, user_text, faq_text)
        try:
            with st.spinner("Thinking..."):
                answer = ask_llm(prompt, system, temperature)
        except RuntimeError as error:
            answer = f"⚠️ {error}"

        bot_msg = {"role": "assistant", "content": answer, "task": task,
                   "technique": technique, "system": system, "prompt": prompt}
        show_message(bot_msg)
        st.session_state.messages.append(bot_msg)

# ------------------------------------------------ Tab 2: Compare prompts
with tab_compare:
    st.subheader("Compare prompt techniques side by side")
    st.caption("The same input is sent using all four techniques so you can see the difference.")

    cmp_task = st.selectbox("Task", TASKS, key="cmp_task")
    cmp_input = st.text_area("Input", value=SAMPLE_INPUTS[cmp_task], height=130,
                             key=f"cmp_input_{cmp_task}")

    if st.button("Run comparison", type="primary"):
        columns = st.columns(len(TECHNIQUES))
        table = []
        try:
            for col, tech in zip(columns, TECHNIQUES):
                system, prompt = build_prompt(cmp_task, tech, cmp_input, faq_text)
                start = time.time()
                with col:
                    with st.spinner(f"{tech}..."):
                        answer = ask_llm(prompt, system, temperature)
                    st.markdown(f"**{tech}**")
                    st.write(answer)
                row = {"Technique": tech, "Words": len(answer.split()),
                       "Time (s)": round(time.time() - start, 1)}
                if cmp_task == "Summarization":
                    row["Compression"] = round(len(answer.split()) / max(len(cmp_input.split()), 1), 2)
                table.append(row)
            st.markdown("#### Quick comparison")
            st.dataframe(pd.DataFrame(table), hide_index=True)
        except RuntimeError as error:
            st.error(str(error))

# ------------------------------------------------------ Tab 3: Evaluation
with tab_eval:
    st.subheader("Evaluate quality and consistency")
    st.caption("Runs the test queries (data/test_queries.csv) through each technique, several times, and scores the answers.")

    eval_task = st.radio("Task to evaluate", ["Q&A", "Classification"], horizontal=True)
    eval_techniques = st.multiselect("Techniques", TECHNIQUES, default=TECHNIQUES)

    # Q&A needs expected keywords, so rows without keywords are skipped
    eval_pool = test_df.dropna(subset=["expected_keywords"]) if eval_task == "Q&A" else test_df

    col1, col2 = st.columns(2)
    n_queries = col1.slider("Number of test queries", 2, len(eval_pool), min(4, len(eval_pool)))
    runs = col2.slider("Runs per query (for consistency)", 1, 3, 2)

    calls = n_queries * len(eval_techniques) * runs
    st.info(f"This will make about **{calls} API calls**. The free tier has rate limits, so keep this number small.")

    if st.button("Run evaluation", type="primary", disabled=not eval_techniques):
        sample = eval_pool.sample(n_queries, random_state=1)
        progress = st.progress(0.0, text="Running evaluation...")
        try:
            results = run_evaluation(eval_task, sample, eval_techniques, faq_text, runs, temperature,
                                     on_progress=lambda p: progress.progress(p, text="Running evaluation..."))
            st.session_state.eval_results = (eval_task, results)
        except RuntimeError as error:
            st.error(str(error))
        progress.empty()

    if "eval_results" in st.session_state:
        done_task, results = st.session_state.eval_results
        summary = summarize_results(results)

        st.markdown(f"#### Results for {done_task}")
        quality_name = "Keyword score" if done_task == "Q&A" else "Accuracy"
        st.caption(f"score = {quality_name} (quality) | consistency = how similar repeated answers are "
                   "| format_ok = reply was only the category name | words = average answer length")
        st.dataframe(summary)
        st.bar_chart(summary[["score", "consistency"]])

        if "Zero-shot" in summary.index and "Improved" in summary.index:
            gain = summary.loc["Improved", "score"] - summary.loc["Zero-shot", "score"]
            st.success(f"Improved prompt vs Zero-shot: quality score changed by {gain:+.2f}")

        with st.expander("See detailed results"):
            st.dataframe(results)
        st.download_button("Download results (CSV)", results.to_csv(index=False),
                           file_name="evaluation_results.csv", mime="text/csv")

# --------------------------------------------------------- Tab 4: Dataset
with tab_data:
    st.subheader("FAQ dataset (data/faq.csv)")
    st.dataframe(faq_df, hide_index=True)
    st.bar_chart(faq_df["category"].value_counts())

    st.subheader("Test queries (data/test_queries.csv)")
    st.dataframe(test_df, hide_index=True)
