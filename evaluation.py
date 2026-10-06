"""
evaluation.py - Simple ways to measure response quality and consistency.

Quality
  Q&A            : keyword score  = share of expected keywords found in the answer
  Classification : accuracy       = predicted category == expected category
                   format_ok      = the reply was ONLY the category name

Consistency (same prompt asked several times)
  Q&A            : average text similarity between the repeated answers (0 to 1)
  Classification : share of runs that gave the most common label (0 to 1)
"""

import re
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd

from llm import ask_llm
from prompts import CATEGORIES, TECHNIQUES, build_prompt


def keyword_score(response, keywords):
    """Share (0 to 1) of the expected keywords (separated by |) found in the response."""
    words = [k.strip().lower().replace(",", "") for k in str(keywords).split("|") if k.strip()]
    if not words:
        return 0.0
    text = response.lower().replace(",", "")
    return sum(1 for w in words if w in text) / len(words)


def consistency_score(responses):
    """Average similarity (0 to 1) between every pair of responses."""
    if len(responses) < 2:
        return 1.0
    sims = [SequenceMatcher(None, a.lower(), b.lower()).ratio() for a, b in combinations(responses, 2)]
    return sum(sims) / len(sims)


def extract_label(response):
    """Find which category the response mentions. Returns 'Unclear' if none or several."""
    found = [c for c in CATEGORIES if re.search(rf"\b{re.escape(c.lower())}\b", response.lower())]
    return found[0] if len(found) == 1 else "Unclear"


def is_clean_label(response):
    """True if the response is exactly one category name (the format we asked for)."""
    cleaned = response.strip().strip(".*").strip().lower()
    return cleaned in [c.lower() for c in CATEGORIES]


def run_evaluation(task, test_df, techniques, faq_text, runs=2, temperature=0.3, on_progress=None):
    """
    Ask every test query with every technique `runs` times and score the answers.
    Returns a DataFrame with one row per (technique, query).
    """
    rows = []
    total = len(test_df) * len(techniques) * runs
    done = 0

    for technique in techniques:
        for _, row in test_df.iterrows():
            responses = []
            for _ in range(runs):
                system, prompt = build_prompt(task, technique, row["query"], faq_text)
                responses.append(ask_llm(prompt, system, temperature))
                done += 1
                if on_progress:
                    on_progress(done / total)

            result = {"technique": technique, "query": row["query"], "response": responses[0]}

            if task == "Q&A":
                scores = [keyword_score(r, row["expected_keywords"]) for r in responses]
                result["score"] = sum(scores) / runs
                result["consistency"] = consistency_score(responses)
                result["words"] = sum(len(r.split()) for r in responses) / runs
            else:  # Classification
                labels = [extract_label(r) for r in responses]
                expected = row["expected_category"]
                correct = [label.lower() == expected.lower() for label in labels]
                result["expected"] = expected
                result["predicted"] = ", ".join(labels)
                result["score"] = sum(correct) / runs
                result["format_ok"] = sum(is_clean_label(r) for r in responses) / runs
                result["consistency"] = Counter(labels).most_common(1)[0][1] / runs

            rows.append(result)

    return pd.DataFrame(rows)


def summarize_results(results):
    """Average the scores for each technique."""
    columns = [c for c in ["score", "consistency", "format_ok", "words"] if c in results.columns]
    summary = results.groupby("technique")[columns].mean().round(2)
    summary = summary.reindex([t for t in TECHNIQUES if t in summary.index])
    return summary.dropna(axis=1, how="all")
