import io
import json
import re
from typing import Any

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader


# =========================================================
# App configuration
# =========================================================

APP_NAME = "Fit Check"
MODEL_NAME = "gemini-2.5-flash"
MAX_RESUME_CHARS = 60_000
MAX_JOB_CHARS = 40_000
MIN_RESUME_CHARS = 250
MIN_JOB_CHARS = 300

SCORE_WEIGHTS = {
    "keyword_match": 0.30,
    "required_skills_match": 0.25,
    "experience_relevance": 0.20,
    "role_alignment": 0.10,
    "ats_readability": 0.10,
    "education_certifications": 0.05,
}

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


# =========================================================
# Gemini structured-output schema
# =========================================================

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "weaknesses": {
            "type": "array",
            "items": {"type": "string"},
        },
        "score_breakdown": {
            "type": "object",
            "properties": {
                "keyword_match": {"type": "integer", "minimum": 0, "maximum": 100},
                "required_skills_match": {"type": "integer", "minimum": 0, "maximum": 100},
                "experience_relevance": {"type": "integer", "minimum": 0, "maximum": 100},
                "role_alignment": {"type": "integer", "minimum": 0, "maximum": 100},
                "ats_readability": {"type": "integer", "minimum": 0, "maximum": 100},
                "education_certifications": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": [
                "keyword_match",
                "required_skills_match",
                "experience_relevance",
                "role_alignment",
                "ats_readability",
                "education_certifications",
            ],
        },
        "missing_keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "priority": {"type": "string", "enum": ["Critical", "Important", "Nice to Have"]},
                    "why_it_matters": {"type": "string"},
                    "suggested_section": {"type": "string"},
                    "suggested_natural_usage": {"type": "string"},
                    "truthfulness_warning": {"type": "string"},
                    "job_description_count": {"type": "integer", "minimum": 0},
                },
                "required": [
                    "keyword",
                    "priority",
                    "why_it_matters",
                    "suggested_section",
                    "suggested_natural_usage",
                    "truthfulness_warning",
                    "job_description_count",
                ],
            },
        },
        "matched_keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "importance": {"type": "string", "enum": ["Critical", "Important", "Nice to Have"]},
                    "resume_presence": {"type": "string"},
                    "resume_count": {"type": "integer", "minimum": 0},
                    "job_description_count": {"type": "integer", "minimum": 0},
                    "strength": {"type": "string"},
                    "suggested_improvement": {"type": "string"},
                },
                "required": [
                    "keyword",
                    "importance",
                    "resume_presence",
                    "resume_count",
                    "job_description_count",
                    "strength",
                    "suggested_improvement",
                ],
            },
        },
        "weak_keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "importance": {"type": "string", "enum": ["Critical", "Important", "Nice to Have"]},
                    "resume_count": {"type": "integer", "minimum": 0},
                    "job_description_count": {"type": "integer", "minimum": 0},
                    "why_weak": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": [
                    "keyword",
                    "importance",
                    "resume_count",
                    "job_description_count",
                    "why_weak",
                    "recommended_action",
                ],
            },
        },
        "overused_keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "resume_count": {"type": "integer", "minimum": 0},
                    "why_overused": {"type": "string"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["keyword", "resume_count", "why_overused", "recommended_action"],
            },
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    "action": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["priority", "action", "reason"],
            },
        },
        "section_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": [
                            "Contact/Header",
                            "Professional Summary",
                            "Skills",
                            "Experience",
                            "Projects",
                            "Education",
                            "Certifications",
                        ],
                    },
                    "status": {"type": "string", "enum": ["Strong", "Good", "Needs Improvement", "Missing"]},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "problems": {"type": "array", "items": {"type": "string"}},
                    "recommended_changes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["section", "status", "strengths", "problems", "recommended_changes"],
            },
        },
        "action_plan": {
            "type": "array",
            "items": {"type": "string"},
        },
        "potential_gaps_or_red_flags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "summary",
        "strengths",
        "weaknesses",
        "score_breakdown",
        "missing_keywords",
        "matched_keywords",
        "weak_keywords",
        "overused_keywords",
        "recommendations",
        "section_feedback",
        "action_plan",
        "potential_gaps_or_red_flags",
    ],
}


# =========================================================
# Styling
# =========================================================

st.set_page_config(
    page_title="Fit Check — Resume ATS Analyzer",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #f6f9fc;
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(135deg, #0f5bd8 0%, #2e80ff 100%);
            color: white;
            padding: 2.1rem 2.3rem;
            border-radius: 22px;
            box-shadow: 0 16px 40px rgba(25, 93, 210, 0.18);
            margin-bottom: 1.4rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.7rem;
            line-height: 1.1;
            letter-spacing: -0.03em;
        }

        .hero p {
            margin: 0.65rem 0 0;
            font-size: 1.06rem;
            opacity: 0.95;
        }

        .card {
            background: white;
            border: 1px solid #e6edf5;
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 8px 26px rgba(23, 49, 84, 0.06);
            margin-bottom: 1rem;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
        }

        .score-number {
            font-size: 4.2rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.04em;
        }

        .score-caption {
            color: #64748b;
            margin-top: 0.2rem;
        }

        .pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            background: #eef5ff;
            color: #0f5bd8;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e6edf5;
            padding: 0.85rem;
            border-radius: 16px;
            box-shadow: 0 6px 20px rgba(23, 49, 84, 0.05);
        }

        .footer-note {
            margin-top: 2rem;
            color: #64748b;
            font-size: 0.85rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Utility functions
# =========================================================


def clean_text(text: str) -> str:
    """Normalize extracted/pasted text without changing its meaning."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return clean_text("\n\n".join(pages))


def extract_docx(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    chunks: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            chunks.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                chunks.append(" | ".join(cells))

    return clean_text("\n".join(chunks))


def extract_txt(file_bytes: bytes) -> str:
    return clean_text(file_bytes.decode("utf-8", errors="replace"))


def extract_resume(uploaded_file: Any) -> str:
    """Extract resume text based on file extension."""
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
    file_bytes = uploaded_file.getvalue()

    if extension == "pdf":
        return extract_pdf(file_bytes)
    if extension == "docx":
        return extract_docx(file_bytes)
    if extension == "txt":
        return extract_txt(file_bytes)
    raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")


def truncate_for_model(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep input bounded while preserving the beginning and ending context."""
    if len(text) <= max_chars:
        return text, False

    head = int(max_chars * 0.72)
    tail = max_chars - head
    truncated = text[:head] + "\n\n[...middle of document omitted for model input...]\n\n" + text[-tail:]
    return truncated, True


def get_api_key() -> str | None:
    """Read the Gemini key from Streamlit secrets only."""
    try:
        value = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        value = None
    if not value:
        return None
    return str(value).strip()


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    """Create one reusable Gemini client per API key."""
    return genai.Client(api_key=api_key)


def score_category(score: int) -> str:
    if score >= 80:
        return "Excellent Match"
    if score >= 65:
        return "Good Match"
    if score >= 50:
        return "Moderate Match"
    return "Needs Improvement"


def score_emoji(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 65:
        return "🟡"
    if score >= 50:
        return "🟠"
    return "🔴"


def weighted_overall_score(breakdown: dict[str, Any]) -> int:
    """Calculate the overall score locally using the transparent weights."""
    total = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        try:
            value = int(breakdown.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        value = max(0, min(100, value))
        total += value * weight
    return int(round(total))


def safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def validate_analysis(data: Any) -> dict[str, Any]:
    """Apply light validation so the UI never crashes on an imperfect model response."""
    if not isinstance(data, dict):
        raise ValueError("The AI returned an unexpected response format.")

    required = [
        "summary",
        "strengths",
        "weaknesses",
        "score_breakdown",
        "missing_keywords",
        "matched_keywords",
        "weak_keywords",
        "overused_keywords",
        "recommendations",
        "section_feedback",
        "action_plan",
        "potential_gaps_or_red_flags",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"The AI response is missing required sections: {', '.join(missing)}")

    breakdown = safe_dict(data["score_breakdown"])
    for key in SCORE_WEIGHTS:
        value = breakdown.get(key)
        if not isinstance(value, (int, float)):
            breakdown[key] = 0
        breakdown[key] = int(max(0, min(100, breakdown[key])))

    data["score_breakdown"] = breakdown
    data["summary"] = str(data.get("summary", "")).strip()
    data["strengths"] = [str(x) for x in safe_list(data["strengths"])][:10]
    data["weaknesses"] = [str(x) for x in safe_list(data["weaknesses"])][:10]
    data["action_plan"] = [str(x) for x in safe_list(data["action_plan"])][:5]
    data["potential_gaps_or_red_flags"] = [str(x) for x in safe_list(data["potential_gaps_or_red_flags"])][:10]

    for key in ["missing_keywords", "matched_keywords", "weak_keywords", "overused_keywords", "recommendations", "section_feedback"]:
        data[key] = safe_list(data[key])

    return data


def build_prompt(resume_text: str, job_description: str) -> str:
    """Build a deterministic, evidence-first analysis prompt."""
    return f"""
You are Fit Check's resume ATS-style compatibility engine. Act simultaneously as:
- an ATS optimization specialist,
- a technical recruiter,
- a resume reviewer,
- a job-description keyword analyst.

Your task is to compare the candidate resume with the target job description and return ONLY the structured JSON object requested by the response schema.

IMPORTANT EVIDENCE RULES:
1. Use only facts explicitly supported by the resume text. Never invent employers, years, education, certifications, technologies, responsibilities, metrics, achievements, or experience.
2. Distinguish clearly between: present in resume, missing from resume, weakly represented, and recommended only if truthful.
3. Semantic matching is allowed when two terms clearly represent the same skill or concept (for example, "Machine Learning" and "ML"), but do not call unrelated concepts matches.
4. Do not reward keyword stuffing. Natural, contextual usage is better than repeated exact phrases.
5. Keyword counts are simple occurrence estimates in the supplied text. Do not pretend they are proprietary ATS counts.
6. A keyword can be missing from the resume but still be a valid recommendation ONLY when the truthfulness warning explicitly says it must be added only if the candidate genuinely has that skill/experience.
7. Evaluate resume readability for typical ATS parsing: conventional headings, clear dates/job titles, plain readable content, consistent bullets, and low dependence on graphics/tables. You only see extracted text, so do not claim to inspect visual design details that extraction cannot prove.
8. For job title alignment, compare the apparent target role/title in the resume with the target job's role/title and responsibilities.
9. Score each dimension independently from 0 to 100 using evidence from the documents.
10. Keep recommendations concrete and prioritized.

SCORING DIMENSIONS:
- keyword_match: semantic and exact keyword coverage
- required_skills_match: evidence of required/core skills from the job description
- experience_relevance: relevance and strength of experience/projects to the role
- role_alignment: job title, summary, and overall positioning alignment
- ats_readability: likely text-based ATS readability based on extracted content/structure
- education_certifications: alignment of education and certifications with stated job requirements

The overall score will be calculated by the application using these exact weights:
- keyword_match = 30%
- required_skills_match = 25%
- experience_relevance = 20%
- role_alignment = 10%
- ats_readability = 10%
- education_certifications = 5%
Do NOT try to override this weighting.

KEYWORD RULES:
- List the highest-value missing keywords first.
- For each missing keyword, provide the job-description occurrence count and a natural usage target expressed contextually (for example, "use in Skills + 1 relevant bullet if genuinely experienced"), not an arbitrary repetition quota.
- Identify underrepresented/weak keywords when resume usage is noticeably thinner than the job description or lacks context.
- Identify overused keywords only when repetition is plausibly hurting readability; do not label ordinary repetition as overuse.
- Treat spelling variants, abbreviations, and close synonyms intelligently.

SECTION REVIEW:
Review these sections when they are present: Contact/Header, Professional Summary, Skills, Experience, Projects, Education, Certifications. If a section is absent from the extracted resume, mark it Missing rather than guessing.

RETURN CONTENT THAT IS USEFUL TO A BEGINNER. Keep individual recommendation items concise but specific.

=== RESUME TEXT ===
{resume_text}

=== JOB DESCRIPTION ===
{job_description}
""".strip()


def analyze_with_gemini(resume_text: str, job_description: str) -> dict[str, Any]:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Gemini API key is missing. Add GEMINI_API_KEY to Streamlit Secrets.")

    client = get_gemini_client(api_key)
    prompt = build_prompt(resume_text, job_description)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ANALYSIS_SCHEMA,
        temperature=0.2,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )

    raw_text = getattr(response, "text", None)
    if not raw_text:
        raise ValueError("Gemini returned an empty response.")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned malformed JSON. Please try the analysis again.") from exc

    return validate_analysis(data)


# =========================================================
# Session state
# =========================================================

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "resume_meta" not in st.session_state:
    st.session_state.resume_meta = None


# =========================================================
# Header
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>Fit Check</h1>
        <p>See how well your resume fits the job.</p>
        <p>Upload your resume and paste the job description to get an AI-powered ATS-style compatibility analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Sidebar guidance
# =========================================================

with st.sidebar:
    st.markdown("### How Fit Check works")
    st.write("1. Upload a PDF, DOCX, or TXT resume.")
    st.write("2. Paste the complete job description.")
    st.write("3. Run one analysis with Gemini.")
    st.write("4. Review the score, keyword gaps, and action plan.")

    st.divider()
    st.markdown("### Scoring weights")
    st.write("Keyword Match — 30%")
    st.write("Required Skills — 25%")
    st.write("Experience/Relevance — 20%")
    st.write("Role Alignment — 10%")
    st.write("ATS Readability — 10%")
    st.write("Education/Certifications — 5%")

    st.divider()
    st.info(
        "Privacy: the app keeps resume text in the current session only and does not intentionally write resumes to permanent storage. "
        "The extracted text is sent to Gemini for the requested analysis."
    )


# =========================================================
# Input area
# =========================================================

left, right = st.columns([0.95, 1.25], gap="large")

with left:
    st.markdown("### 1. Upload Resume")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=sorted(ALLOWED_EXTENSIONS),
        help="For best results, use a text-based PDF or a DOCX with standard resume sections.",
    )

    resume_text = ""
    if uploaded_file is not None:
        extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
        file_size_kb = len(uploaded_file.getvalue()) / 1024

        st.success(f"Uploaded: **{uploaded_file.name}**")
        st.caption(f"Type: {extension.upper()} · Size: {file_size_kb:.1f} KB")

        try:
            with st.spinner("Extracting resume text…"):
                resume_text = extract_resume(uploaded_file)
        except Exception:
            st.error(
                "Fit Check could not read this file. Please verify that the document is not corrupted "
                "and that a PDF contains selectable text."
            )
            resume_text = ""

        if resume_text:
            word_count = len(resume_text.split())
            st.caption(f"Extracted text: {word_count:,} words")
            if len(resume_text) < MIN_RESUME_CHARS:
                st.warning("The extracted resume is very short. Use a clearer/text-based document before analyzing.")
        elif uploaded_file is not None:
            st.error("No readable text was extracted from the resume.")

with right:
    st.markdown("### 2. Target Job Description")
    job_description = st.text_area(
        "Paste the complete job description",
        height=360,
        placeholder="Paste the full job description here…",
        help="Include the responsibilities, required skills, preferred skills, and qualifications when available.",
    )
    job_description = clean_text(job_description)
    if job_description:
        st.caption(f"Job description length: {len(job_description):,} characters")

st.markdown("### 3. Analyze Resume")

col_a, col_b, col_c = st.columns([1.4, 1, 1])
with col_a:
    analyze_clicked = st.button("Analyze My Resume", type="primary", use_container_width=True)
with col_b:
    if st.button("Clear Results", use_container_width=True):
        st.session_state.analysis = None
        st.session_state.resume_meta = None
        st.rerun()
with col_c:
    st.caption("One Gemini request per button click")


# =========================================================
# Validation + analysis
# =========================================================

if analyze_clicked:
    # Re-extract only when the user clicks the button so normal reruns do not call Gemini.
    if uploaded_file is None:
        st.error("Please upload your resume before analyzing.")
    elif not resume_text:
        st.error("Please upload a readable resume before analyzing.")
    elif len(resume_text) < MIN_RESUME_CHARS:
        st.error("The resume text is too short to analyze reliably. Please use a fuller, readable resume document.")
    elif not job_description:
        st.error("Please paste the target job description before analyzing.")
    elif len(job_description) < MIN_JOB_CHARS:
        st.error("The job description is too short. Paste the complete job description for a meaningful comparison.")
    elif not get_api_key():
        st.error(
            "Gemini API key not found. In Streamlit Community Cloud, open your app's Settings → Secrets "
            "and add GEMINI_API_KEY."
        )
    else:
        model_resume, resume_truncated = truncate_for_model(resume_text, MAX_RESUME_CHARS)
        model_job, job_truncated = truncate_for_model(job_description, MAX_JOB_CHARS)

        with st.spinner("Analyzing resume against the job description…"):
            try:
                analysis = analyze_with_gemini(model_resume, model_job)
                st.session_state.analysis = analysis
                st.session_state.resume_meta = {
                    "filename": uploaded_file.name,
                    "resume_words": len(resume_text.split()),
                    "resume_truncated": resume_truncated,
                    "job_truncated": job_truncated,
                }
            except Exception as exc:
                message = str(exc).strip() or "The analysis could not be completed."
                lowered = message.lower()

                if "api key" in lowered or "authentication" in lowered or "unauthenticated" in lowered:
                    user_message = "Gemini authentication failed. Check that your GEMINI_API_KEY is correct and active."
                elif "quota" in lowered or "rate" in lowered or "resource exhausted" in lowered:
                    user_message = "Gemini rate/quota limit reached. Wait a little and try again, or review your API quota."
                elif "timeout" in lowered or "deadline" in lowered:
                    user_message = "The Gemini request timed out. Please try again with a slightly shorter resume or job description."
                elif "json" in lowered or "response format" in lowered or "unexpected response" in lowered:
                    user_message = "Gemini returned an unexpected response format. Please run the analysis again."
                else:
                    user_message = "Fit Check could not complete the analysis. Please check your inputs and try again."

                st.error(user_message)
                # Details are intentionally kept out of normal user-facing output to avoid exposing internals.
                st.session_state.analysis = None


# =========================================================
# Results dashboard
# =========================================================

analysis = st.session_state.analysis

if analysis:
    meta = st.session_state.resume_meta or {}
    overall = weighted_overall_score(analysis["score_breakdown"])
    category = score_category(overall)

    st.divider()
    st.markdown("## Results Dashboard")

    score_col, summary_col = st.columns([0.8, 1.8], gap="large")

    with score_col:
        st.markdown(
            f"""
            <div class="card">
                <div class="pill">ATS-style estimate</div>
                <div class="score-number">{overall}/100</div>
                <div><strong>{score_emoji(overall)} {category}</strong></div>
                <div class="score-caption">AI-generated compatibility estimate — not an official ATS vendor score.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with summary_col:
        st.markdown("### Executive Summary")
        st.write(f"**Your resume is currently a {overall}/100 match for this role.**")
        st.write(analysis["summary"])

    if meta.get("resume_truncated") or meta.get("job_truncated"):
        st.info(
            "The uploaded text was longer than the model-input safety limit, so the model received the beginning and ending portions of the text."
        )

    st.markdown("### Score Breakdown")
    b = analysis["score_breakdown"]
    metric_cols = st.columns(6)
    metric_labels = [
        ("Keyword Match", b["keyword_match"]),
        ("Required Skills", b["required_skills_match"]),
        ("Experience Relevance", b["experience_relevance"]),
        ("Role Alignment", b["role_alignment"]),
        ("ATS Readability", b["ats_readability"]),
        ("Education / Certs", b["education_certifications"]),
    ]
    for col, (label, value) in zip(metric_cols, metric_labels):
        with col:
            col.metric(label, f"{value}/100")
            st.progress(value / 100)

    st.markdown("### Biggest Strengths")
    if analysis["strengths"]:
        for strength in analysis["strengths"][:6]:
            st.success(strength, icon="✓")
    else:
        st.write("No strengths were returned.")

    st.markdown("### Biggest Weaknesses")
    if analysis["weaknesses"]:
        for weakness in analysis["weaknesses"][:6]:
            st.warning(weakness, icon="!")
    else:
        st.write("No major weaknesses were returned.")

    # -----------------------------------------------------
    # Keyword intelligence
    # -----------------------------------------------------
    st.divider()
    st.markdown("## Keyword Intelligence")

    tab_missing, tab_matched, tab_weak, tab_overused = st.tabs(
        ["Missing Keywords", "Matched Keywords", "Weak / Underused", "Potential Overuse"]
    )

    with tab_missing:
        rows = []
        for item in analysis["missing_keywords"]:
            rows.append(
                {
                    "Keyword": item.get("keyword", ""),
                    "Priority": item.get("priority", ""),
                    "Resume Count": 0,
                    "JD Count": item.get("job_description_count", 0),
                    "Why It Matters": item.get("why_it_matters", ""),
                    "Suggested Section": item.get("suggested_section", ""),
                    "Natural Usage": item.get("suggested_natural_usage", ""),
                    "Truthfulness": item.get("truthfulness_warning", ""),
                }
            )
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.success("No major missing keywords were identified.")

    with tab_matched:
        rows = []
        for item in analysis["matched_keywords"]:
            rows.append(
                {
                    "Keyword": item.get("keyword", ""),
                    "Importance": item.get("importance", ""),
                    "Resume Count": item.get("resume_count", 0),
                    "JD Count": item.get("job_description_count", 0),
                    "Presence": item.get("resume_presence", ""),
                    "Strength": item.get("strength", ""),
                    "Suggested Improvement": item.get("suggested_improvement", ""),
                }
            )
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No matched keywords were returned.")

    with tab_weak:
        rows = []
        for item in analysis["weak_keywords"]:
            rows.append(
                {
                    "Keyword": item.get("keyword", ""),
                    "Importance": item.get("importance", ""),
                    "Resume Count": item.get("resume_count", 0),
                    "JD Count": item.get("job_description_count", 0),
                    "Why Weak": item.get("why_weak", ""),
                    "Recommended Action": item.get("recommended_action", ""),
                }
            )
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.success("No obvious underrepresented keywords were identified.")

    with tab_overused:
        rows = []
        for item in analysis["overused_keywords"]:
            rows.append(
                {
                    "Keyword": item.get("keyword", ""),
                    "Resume Count": item.get("resume_count", 0),
                    "Why Overused": item.get("why_overused", ""),
                    "Recommended Action": item.get("recommended_action", ""),
                }
            )
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.success("No meaningful keyword overuse was identified.")

    st.info(
        "Keyword counts are simple occurrence estimates from the supplied text. Fit Check is designed to improve truthful, natural alignment—not to encourage keyword stuffing."
    )

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------
    st.divider()
    st.markdown("## Prioritized Improvements")

    recommendations = analysis["recommendations"]
    priority_order = ["High", "Medium", "Low"]
    for priority in priority_order:
        items = [item for item in recommendations if item.get("priority") == priority]
        if not items:
            continue
        st.markdown(f"### {priority} Priority")
        for item in items:
            st.markdown(f"**{item.get('action', 'Recommendation')}**")
            st.write(item.get("reason", ""))

    # -----------------------------------------------------
    # Section feedback
    # -----------------------------------------------------
    st.divider()
    st.markdown("## Section-by-Section Feedback")

    section_feedback = analysis["section_feedback"]
    if section_feedback:
        for item in section_feedback:
            status = item.get("status", "")
            title = f"{item.get('section', 'Section')} — {status}"
            with st.expander(title, expanded=False):
                strengths = safe_list(item.get("strengths"))
                problems = safe_list(item.get("problems"))
                changes = safe_list(item.get("recommended_changes"))

                if strengths:
                    st.markdown("**Strengths**")
                    for value in strengths:
                        st.write(f"• {value}")
                if problems:
                    st.markdown("**Problems**")
                    for value in problems:
                        st.write(f"• {value}")
                if changes:
                    st.markdown("**Recommended changes**")
                    for value in changes:
                        st.write(f"• {value}")
    else:
        st.info("No section feedback was returned.")

    # -----------------------------------------------------
    # Gaps / red flags
    # -----------------------------------------------------
    if analysis["potential_gaps_or_red_flags"]:
        st.divider()
        st.markdown("## Potential Gaps or Red Flags")
        for item in analysis["potential_gaps_or_red_flags"]:
            st.warning(item, icon="⚠️")

    # -----------------------------------------------------
    # Action plan
    # -----------------------------------------------------
    st.divider()
    st.markdown("## Recommended Action Plan")
    st.write("Before applying, make these changes first:")
    for index, action in enumerate(analysis["action_plan"], start=1):
        st.markdown(f"**{index}.** {action}")

    st.divider()
    st.markdown(
        "<div class='footer-note'>Fit Check provides an AI-generated ATS-style compatibility estimate. "
        "It is not an official score from Workday, Greenhouse, Taleo, Lever, or any other ATS provider, and a higher score does not guarantee an interview.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='card'><strong>Ready when you are.</strong><br>Upload a resume and paste a complete job description, then click <em>Analyze My Resume</em>.</div>",
        unsafe_allow_html=True,
    )
