import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq


load_dotenv()


def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")

    if key:
        return key

    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def answer_manual_question(question, evidence):
    api_key = get_groq_api_key()

    if not api_key:
        return (
            "Groq API key is missing. Configure GROQ_API_KEY "
            "in your local .env file or deployment secrets."
        )

    if not evidence:
        return "No relevant manual passages were found."

    evidence_text = ""

    for item in evidence:
        evidence_text += (
            f"\nSource: {item['source']}\n"
            f"Page: {item['page']}\n"
            f"Passage: {item['text']}\n"
        )

    prompt = f"""
You are an electrical maintenance documentation assistant.

Question:
{question}

Evidence from the uploaded equipment manual:
{evidence_text}

Answer using the provided evidence only.

Include:
1. A direct answer in simple language.
2. Relevant checks mentioned in the manual.
3. Source filenames and page numbers.

If the evidence is insufficient, say so clearly.
Do not invent manufacturer instructions or safety limits.

This answer is informational support, not authorization
to perform electrical work.
"""

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            reasoning_effort="low",
        )

        return response.choices[0].message.content

    except Exception:
        return (
            "The explanation could not be generated. "
            "Check your Groq API configuration and try again."
        )