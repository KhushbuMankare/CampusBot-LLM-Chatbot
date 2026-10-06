"""
llm.py - Talks to the Gemini API (Google Gen AI SDK).

All other files call ask_llm() so the API code lives in only one place.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Read variables from the .env file into the environment
load_dotenv()

# Model can be changed from the .env file without touching the code
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def api_key_available():
    """Return True if GEMINI_API_KEY is set in the .env file / environment."""
    return bool(os.getenv("GEMINI_API_KEY"))


def ask_llm(prompt, system_instruction=None, temperature=0.3, retries=3):
    """
    Send a prompt to Gemini and return the response text.

    prompt             : the main user prompt (string)
    system_instruction : optional role / rules for the model
    temperature        : 0.0 = very predictable, 1.0 = more creative
    retries            : number of attempts for temporary API errors
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
    )

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )

            return (response.text or "").strip()

        except Exception as error:
            message = str(error)

            # Temporary errors that are worth retrying
            temporary_error = (
                "429" in message
                or "RESOURCE_EXHAUSTED" in message
                or "503" in message
                or "UNAVAILABLE" in message
            )

            if temporary_error and attempt < retries:

                # Exponential backoff:
                # attempt 1 -> wait 5 seconds
                # attempt 2 -> wait 10 seconds
                wait_time = 5 * (2 ** (attempt - 1))

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            raise RuntimeError(
                f"Gemini API error: {message}"
            ) from error