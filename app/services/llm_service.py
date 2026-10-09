import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


def answer_manual_question(question, evidence):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "Groq API key is missing. Add GROQ_API_KEY "
            "to your .env file to enable explanations."
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

User question:
{question}

Evidence retrieved from equipment manuals:
{evidence_text}

Answer using the supplied evidence only.

Include:
1. A direct answer in simple language
2. Relevant inspection checks mentioned in the evidence
3. The source filename and page number for important claims

If the evidence does not answer the question, say so clearly.
Do not invent manufacturer instructions, equipment limits,
or safety procedures.

Treat your answer as informational support, not a confirmed
diagnosis or authorization to perform electrical work.
Recommend a qualified electrical professional where appropriate.
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

    except Exception as error:
        return f"Could not generate the explanation: {error}"