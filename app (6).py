
import os
import json
import re
import streamlit as st
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-120b"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AwaamiAgent",
    page_icon="🏛️",
    layout="centered"
)


# ============================================================
# API KEY
# ============================================================

def get_api_key():
    """
    Get the Groq API key from:
    1. Streamlit Secrets during deployment
    2. Environment variable during Colab/local development
    """

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    return api_key


api_key = get_api_key()

if not api_key:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to the environment or Streamlit Secrets."
    )
    st.stop()


client = Groq(api_key=api_key)


# ============================================================
# AI ANALYSIS
# ============================================================

def analyze_civic_problem(problem, language):
    """
    Analyze a user's civic problem and return structured JSON.
    """

    language_instruction = (
        "Respond in English."
        if language == "English"
        else "Respond in natural, simple Urdu. Keep important English terms "
             "in parentheses when useful."
    )

    prompt = f"""
You are AwaamiAgent, an AI civic assistance system.

Your job is to help an ordinary person understand a civic problem
and identify practical next steps.

{language_instruction}

IMPORTANT SAFETY RULES:
- Do not invent laws.
- Do not invent government departments.
- Do not invent deadlines.
- Do not invent fees.
- Do not invent procedures.
- Do not claim uncertain information is verified.
- If jurisdiction-specific information is missing, clearly say so.
- Do not present the response as legal advice.
- Give general practical guidance only.

Return ONLY valid JSON using exactly these six fields:

{{
  "issue_category": "short category",
  "explanation": "simple explanation of the problem",
  "important_information": [
    "important point 1",
    "important point 2"
  ],
  "next_steps": [
    "practical step 1",
    "practical step 2"
  ],
  "required_documents": [
    "document or information 1",
    "document or information 2"
  ],
  "complaint": "short initial complaint/application draft"
}}

Keep the response concise.

User's civic problem:
{problem}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    raw_output = response.choices[0].message.content.strip()

    return parse_json_response(raw_output)


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(text):
    """
    Parse JSON even if the model accidentally surrounds it
    with markdown code fences.
    """

    # Remove markdown code fences if present
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to locate the JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("The AI returned an invalid JSON response.")

        try:
            result = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            raise ValueError("The AI returned an invalid JSON response.")

    required_fields = [
        "issue_category",
        "explanation",
        "important_information",
        "next_steps",
        "required_documents",
        "complaint"
    ]

    for field in required_fields:
        if field not in result:
            result[field] = []

    return result


# ============================================================
# SEPARATE COMPLAINT GENERATOR
# ============================================================

def generate_complaint(
    issue_category,
    explanation,
    important_information,
    language
):
    """
    Generate a more polished complaint/application
    without inventing personal facts or legal information.
    """

    language_instruction = (
        "Write in English."
        if language == "English"
        else "Write in natural, simple Urdu."
    )

    prompt = f"""
You are AwaamiAgent's complaint drafting assistant.

{language_instruction}

Create a professional complaint/application based ONLY on the
information provided below.

IMPORTANT:
- Do not invent names.
- Do not invent account numbers.
- Do not invent addresses.
- Do not invent dates.
- Do not invent laws.
- Do not invent government departments.
- Do not invent fees or deadlines.
- Use placeholders where personal information is missing.
- Do not provide legal advice.
- Keep the complaint concise and practical.

Issue:
{issue_category}

Explanation:
{explanation}

Important information:
{json.dumps(important_information, ensure_ascii=False)}

Use appropriate placeholders such as:
[Your Name]
[Your Address]
[Account/Reference Number]
[Date]

Return only the complaint/application text.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# ============================================================
# SESSION STATE
# ============================================================

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "complaint" not in st.session_state:
    st.session_state.complaint = None


# ============================================================
# UI
# ============================================================

st.title("🏛️ AwaamiAgent")

st.markdown(
    """
**Your AI civic assistant**

Describe a civic problem in simple words.  
AwaamiAgent will help you understand the issue and identify practical next steps.
"""
)

st.info(
    "AwaamiAgent provides general civic assistance. "
    "Jurisdiction-specific laws, procedures, deadlines, fees, and departments "
    "should be verified through reliable official sources."
)


# ============================================================
# INPUT
# ============================================================

problem = st.text_area(
    "Describe your civic problem",
    placeholder=(
        "Example: My electricity bill is much higher than usual "
        "and I do not understand why."
    ),
    height=160
)

language = st.selectbox(
    "Response language",
    ["English", "Urdu"]
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔎 Analyze Problem", type="primary"):

    if not problem.strip():
        st.warning("Please describe your civic problem first.")
    else:

        st.session_state.analysis = None
        st.session_state.complaint = None

        with st.spinner("AwaamiAgent is analyzing your problem..."):

            try:
                result = analyze_civic_problem(
                    problem.strip(),
                    language
                )

                st.session_state.analysis = result

            except Exception as e:
                st.error(
                    "Something went wrong while analyzing the problem."
                )

                st.caption(f"Technical error: {str(e)}")


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.analysis:

    result = st.session_state.analysis

    st.divider()

    st.subheader("📌 Issue Identified")
    st.write(result.get("issue_category", "Not specified"))

    st.subheader("💡 What's happening?")
    st.write(result.get("explanation", "No explanation available."))

    st.subheader("🔎 Important Information")

    important_information = result.get(
        "important_information",
        []
    )

    if important_information:
        for item in important_information:
            st.write(f"• {item}")
    else:
        st.write("No additional information provided.")

    st.subheader("🧭 What should I do?")

    next_steps = result.get(
        "next_steps",
        []
    )

    if next_steps:
        for index, step in enumerate(next_steps, 1):
            st.write(f"**{index}.** {step}")
    else:
        st.write("No next steps provided.")

    st.subheader("📄 Documents / Information You May Need")

    required_documents = result.get(
        "required_documents",
        []
    )

    if required_documents:
        for item in required_documents:
            st.write(f"• {item}")
    else:
        st.write("No specific documents identified.")

    st.subheader("📝 Initial Complaint / Application")

    st.text_area(
        "AI-generated draft",
        value=result.get("complaint", ""),
        height=250,
        disabled=True
    )

    st.divider()

    st.subheader("✍️ Generate a Polished Complaint")

    if st.button("Generate Complaint"):

        with st.spinner("Generating complaint..."):

            try:
                complaint = generate_complaint(
                    result.get("issue_category", ""),
                    result.get("explanation", ""),
                    important_information,
                    language
                )

                st.session_state.complaint = complaint

            except Exception as e:
                st.error(
                    "Something went wrong while generating the complaint."
                )

                st.caption(f"Technical error: {str(e)}")


# ============================================================
# DISPLAY POLISHED COMPLAINT
# ============================================================

if st.session_state.complaint:

    st.subheader("📄 Polished Complaint / Application")

    st.text_area(
        "Your draft",
        value=st.session_state.complaint,
        height=400
    )

    st.download_button(
        label="⬇️ Download Complaint as TXT",
        data=st.session_state.complaint,
        file_name="awaamiagent_complaint.txt",
        mime="text/plain"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AwaamiAgent is an AI civic assistance tool and is not a government "
    "authority or a substitute for professional legal advice."
)
