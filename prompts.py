"""
prompts.py - All prompt templates in one place.

3 tasks      : Q&A, Summarization, Classification
4 techniques : Zero-shot, Few-shot, Role-based, Improved

"Improved" = role-based + few-shot + clear rules. It is the result of
improving the basic prompts after looking at the evaluation results.
"""

COLLEGE = "Sunrise Institute of Technology"

TASKS = ["Q&A", "Summarization", "Classification"]
TECHNIQUES = ["Zero-shot", "Few-shot", "Role-based", "Improved"]
CATEGORIES = ["Admissions", "Fees", "Exams", "Library", "Hostel", "Placement", "IT Support", "Other"]

TECHNIQUE_INFO = {
    "Zero-shot": "Only the instruction. No examples, no role.",
    "Few-shot": "Instruction + a few solved examples.",
    "Role-based": "The model is told who it is (a system role).",
    "Improved": "Role + examples + strict rules for the output format.",
}

# ---------- Roles (used for role-based prompting) ----------
ROLE_QA = (
    f"You are CampusBot, a friendly and professional help-desk assistant "
    f"for {COLLEGE}. You answer student questions politely and clearly."
)
ROLE_SUMMARY = "You are an expert editor who writes short, accurate and clear summaries."
ROLE_CLASSIFY = f"You are an expert at sorting student queries for the {COLLEGE} help desk."

# ---------- Examples (used for few-shot prompting) ----------
QA_EXAMPLES = """Example 1
Question: What is the passing mark?
Answer: The passing mark is 40 percent in each subject.

Example 2
Question: Does the college have a swimming pool?
Answer: Sorry, I don't have that information. Please contact the college office."""

SUMMARY_EXAMPLE = """Text: The library will stay open until 11 PM during the exam month so that students can prepare. Extra seating has been arranged in the reading hall and a silent zone has been created on the second floor. Students must carry their ID cards and food is not allowed inside.
Summary:
- The library stays open until 11 PM during the exam month.
- Extra seating and a silent zone on the second floor are available.
- ID cards are required and food is not allowed."""

CLASSIFY_EXAMPLES = [
    ("Where do I upload my 12th marksheet for admission?", "Admissions"),
    ("Is there any discount on tuition for toppers?", "Fees"),
    ("When will the hall tickets be released?", "Exams"),
    ("Which shelf has books on data structures?", "Library"),
    ("Can I get a single room in the boys hostel?", "Hostel"),
    ("How do I prepare for the TCS interview?", "Placement"),
    ("The college email login is showing an error", "IT Support"),
    ("Who won the cricket match yesterday?", "Other"),
]


def faq_to_text(faq_df):
    """Turn the FAQ DataFrame into plain text that can be placed inside a prompt."""
    blocks = [f"Q: {row['question']}\nA: {row['answer']}" for _, row in faq_df.iterrows()]
    return "\n\n".join(blocks)


def build_prompt(task, technique, user_input, faq_text=""):
    """Return (system_instruction, prompt) for the chosen task and technique."""
    if task == "Q&A":
        return _qa_prompt(technique, user_input, faq_text)
    if task == "Summarization":
        return _summary_prompt(technique, user_input)
    return _classify_prompt(technique, user_input)


# ---------------------------------------------------------------- Q&A
def _qa_prompt(technique, question, faq_text):
    faq_block = f"FAQ:\n{faq_text}\n\n"

    if technique == "Zero-shot":
        return None, f"{faq_block}Answer the question using the FAQ above.\nQuestion: {question}\nAnswer:"

    if technique == "Few-shot":
        prompt = (
            f"{faq_block}Answer questions using the FAQ above. Here are some examples:\n\n"
            f"{QA_EXAMPLES}\n\nNow answer this one.\nQuestion: {question}\nAnswer:"
        )
        return None, prompt

    if technique == "Role-based":
        return ROLE_QA, f"{faq_block}Question: {question}"

    # Improved
    system = ROLE_QA + (
        "\n\nRules:\n"
        "1. Use ONLY the FAQ. Do not guess or add outside information.\n"
        "2. Keep numbers, dates and timings exactly as written in the FAQ.\n"
        "3. Answer in 1 to 3 short sentences.\n"
        "4. If the answer is not in the FAQ, reply exactly: "
        "\"Sorry, I don't have that information. Please contact the college office.\""
    )
    prompt = f"{faq_block}Examples:\n\n{QA_EXAMPLES}\n\nQuestion: {question}\nAnswer:"
    return system, prompt


# ------------------------------------------------------ Summarization
def _summary_prompt(technique, text):
    if technique == "Zero-shot":
        return None, f"Summarize the following text:\n\n{text}"

    if technique == "Few-shot":
        prompt = (
            f"Summarize texts like in this example.\n\n{SUMMARY_EXAMPLE}\n\n"
            f"Text: {text}\nSummary:"
        )
        return None, prompt

    if technique == "Role-based":
        return ROLE_SUMMARY, f"Summarize the following text:\n\n{text}"

    # Improved
    system = ROLE_SUMMARY + (
        "\n\nRules:\n"
        "1. Write exactly 3 bullet points.\n"
        "2. Each bullet must be under 20 words.\n"
        "3. Keep important facts, names and numbers.\n"
        "4. Do not add opinions or information that is not in the text."
    )
    prompt = f"Example:\n\n{SUMMARY_EXAMPLE}\n\nNow summarize this.\nText: {text}\nSummary:"
    return system, prompt


# ----------------------------------------------------- Classification
def _classify_prompt(technique, query):
    labels = ", ".join(CATEGORIES)
    examples = "\n".join(f"Query: {q}\nCategory: {c}\n" for q, c in CLASSIFY_EXAMPLES)

    if technique == "Zero-shot":
        return None, f"Classify this student query into one of these categories: {labels}.\n\nQuery: {query}"

    if technique == "Few-shot":
        prompt = (
            f"Classify student queries into one of these categories: {labels}.\n\n"
            f"{examples}\nQuery: {query}\nCategory:"
        )
        return None, prompt

    if technique == "Role-based":
        return ROLE_CLASSIFY, f"Categories: {labels}\n\nQuery: {query}"

    # Improved
    system = ROLE_CLASSIFY + (
        "\n\nRules:\n"
        "1. Reply with ONLY the category name, exactly as written in the list.\n"
        "2. No explanation and no extra punctuation.\n"
        "3. If the query is not about the college, reply: Other"
    )
    prompt = f"Categories: {labels}\n\n{examples}\nQuery: {query}\nCategory:"
    return system, prompt
