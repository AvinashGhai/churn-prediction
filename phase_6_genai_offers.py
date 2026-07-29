import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = "openai/gpt-oss-120b"

client = None
api_key = os.environ.get("GROQ_API_KEY")
if api_key:
    client = Groq(api_key=api_key)


def build_prompt(row):
    return f"""You are writing a short, warm retention email for a telecom customer at risk of cancelling.

Customer profile:
- Contract type: {row['contract']}
- Internet service: {row['internetservice']}
- Tenure: {row['tenure']} months
- Monthly charge: ${row['monthlycharges']}
- Churn risk reasons: {row['risk_reasons']}
- Rule-based offer to weave in naturally: {row['recommended_offer']}

Write a 3-4 sentence email body. Be specific to their situation, not generic.
Do not include a subject line or greeting salutation, just the body text."""


def generate_email(row):
    if client is None:
        return f"[Fallback - no API key set] Standard offer: {row['recommended_offer']}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=500,
            reasoning_effort="low",
            messages=[{"role": "user", "content": build_prompt(row)}]
        )
        text = response.choices[0].message.content.strip()
        if not text:
            return f"[Fallback - empty response, reasoning likely consumed the token budget] Standard offer: {row['recommended_offer']}"
        return text
    except Exception as e:
        return f"[Fallback - API error: {e}] Standard offer: {row['recommended_offer']}"


def main():
    top_customers = pd.read_csv("tableau_data/09_top_customers_to_save.csv")

    print(f"Generating personalized retention emails for {len(top_customers)} customers...")

    emails = []
    for i, row in top_customers.iterrows():
        email_text = generate_email(row)
        emails.append(email_text)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(top_customers)} done")

    top_customers["ai_generated_message"] = emails
    top_customers.to_csv("tableau_data/12_genai_retention_emails.csv", index=False)

    print("Saved tableau_data/12_genai_retention_emails.csv")


if __name__ == "__main__":
    main()