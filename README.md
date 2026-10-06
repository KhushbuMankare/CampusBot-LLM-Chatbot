# 🎓 CampusBot: LLM Chatbot with Prompt Engineering

## Project Overview

CampusBot is a simple college help-desk chatbot built with a Large Language Model (Google Gemini) and a Streamlit web interface. The goal of the project is to **show how the way we write a prompt changes the quality of an LLM's answer**.

The bot can do three tasks:

1. **Q&A** – answers student questions using a custom FAQ dataset (CSV).
2. **Summarization** – summarizes a paragraph of text.
3. **Classification** – sorts a student query into a category (Admissions, Fees, Exams, Library, Hostel, Placement, IT Support, Other).

Each task can be run with four prompt styles. The project then **evaluates** the answers and **compares** the techniques, so you can see which prompt works best.

## Features

- Chat interface with task and prompt-technique selection
- Prompt techniques:
  - **Zero-shot** – only the instruction
  - **Few-shot** – instruction + solved examples
  - **Role-based** – the model is given a role (system instruction)
  - **Improved** – role + examples + strict rules (the "engineered" prompt)
- "Show prompt used" option under every bot reply
- Side-by-side prompt comparison tab
- Evaluation tab for quality and consistency, with a results table, bar chart and CSV download
- Custom FAQ dataset (22 questions) and test-query dataset (14 queries)
- Temperature control

### How evaluation works

| Task | Quality | Consistency (same prompt run several times) |
|------|---------|---------------------------------------------|
| Q&A | **Keyword score** – share of expected keywords found in the answer | Average text similarity between repeated answers |
| Classification | **Accuracy** – predicted category equals expected category (also `format_ok`: reply was only the category name) | Share of runs that gave the same label |

The **Improved** prompt is designed to fix common problems of simple prompts (long answers, extra explanations, guessing). Run the evaluation tab to compare it with Zero-shot.

## Technologies Used

- Python 3.9+
- Streamlit (user interface)
- Google Gemini API via the `google-genai` SDK (LLM)
- Pandas (CSV datasets and result tables)
- python-dotenv (loads the API key from `.env`)

## Project Structure

```
chatbot_project/
├── app.py              # Streamlit interface (4 tabs)
├── llm.py              # Gemini API call
├── prompts.py          # Prompt templates for each task and technique
├── evaluation.py       # Quality and consistency scoring
├── data/
│   ├── faq.csv         # FAQ dataset (category, question, answer)
│   └── test_queries.csv# Test queries with expected category and keywords
├── requirements.txt
└── README.md
```

## Installation Steps

1. Make sure Python 3.9 or newer is installed.
2. Open a terminal in the project folder (`chatbot_project`).
3. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac / Linux
   source venv/bin/activate
   ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## API Key Setup

1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and create a free Gemini API key.
2. In the project folder (next to `app.py`), create a file named `.env`.
3. Add this line to it:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
4. *(Optional)* To use a different Gemini model, add:
   ```
   GEMINI_MODEL=gemini-2.5-flash
   ```
   If Google retires this model name in the future, put the name of a current Flash model here.

> Never share your `.env` file or upload it to GitHub. The key is not written anywhere in the code.

## How to Run the Project

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Example Usage

**1. Chatbot tab**
- Choose **Q&A** and **Improved** in the sidebar, then type: `What is the minimum attendance for exams?`
  → The bot answers: attendance must be at least 75 percent.
- Choose **Classification** and type: `My student portal password is not working` → `IT Support`
- Choose **Summarization**, paste a paragraph → you get a 3-bullet summary.
- Click **Show prompt used** under a reply to see the exact prompt sent to the LLM.

**2. Compare Prompts tab**
- Pick a task, keep the sample input (or type your own) and click **Run comparison**.
- You will see the answers from all four techniques next to each other, with word count and time.

**3. Evaluation tab**
- Choose **Classification**, keep all four techniques, 4 queries and 2 runs, and click **Run evaluation**.
- Check the summary table and bar chart. Zero-shot often answers with full sentences (low `format_ok`), while Improved replies with only the category name.
- Try changing a prompt in `prompts.py`, run the evaluation again and compare the scores. This is the prompt-improvement loop.

**4. Dataset tab**
- View the FAQ and test queries. You can edit the CSV files to add your own questions.

## Notes

- The free Gemini tier has rate limits. The app retries automatically, but if you see errors, reduce the number of queries/runs in the Evaluation tab.
- The FAQ is placed directly in the prompt for Q&A (no vector database), which is enough for a small dataset.
- Each chat message is answered independently (no conversation memory) to keep the project simple.
