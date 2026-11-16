from NLP.resume_evidence import extract_listed_skills, find_unlisted_requirements
def is_closing_question(question):
    q = str(question).lower().strip().rstrip("?.! ")
    patterns = (
        "any questions for me",
        "do you have any questions",
        "is there anything you would like to ask",
        "anything else you would like to add",
        "any final thoughts",
        "thank you for your time",
        "before we wrap up",
        "anything about your experience you would like me to know",
        "any question for me",
    )
    return any(p in q for p in patterns)


def build_profile_context(profile):
    if not isinstance(profile, dict):
        return ""
    resume = profile.get("resume", {}) or {}
    job = profile.get("job", {}) or {}
    parts = ["CANDIDATE RESUME EVIDENCE:"]
    summary = str(resume.get("summary", "")).strip()
    if summary:
        parts.append("RESUME SUMMARY (candidate-provided claims; not independently verified): " + summary)
    projects = resume.get("projects", []) or []
    main_project = next((x for x in projects if isinstance(x, dict) and str(x.get("description", "")).count("▸") >= 3), None)
    if main_project:
        parts.append("RESUME PROJECT PORTFOLIO (original extracted text; may contain multiple project descriptions):\n" + str(main_project.get("description", "")).strip())
    else:
        parts.append("No reliable detailed project description was extracted.")
    parts.append("Do not use parser-generated project fragments or inferred skill-to-project links as evidence.")
    parts.append("\nJOB EXPECTATIONS (separate from candidate evidence; never claim these are candidate skills):")
    parts.append("JOB TITLE: " + str(job.get("title", "")))
    reqs = job.get("requirements", {}) or {}
    names = [str(v.get("name", k)) for k, v in reqs.items() if isinstance(v, dict)] if isinstance(reqs, dict) else [str(x) for x in reqs]
    if names:
        parts.append("JOB REQUIREMENTS: " + ", ".join(names))
    listed_skills = extract_listed_skills(resume)
    if listed_skills:
        parts.append("EXPLICITLY LISTED RESUME SKILLS (listed in the skills section; not independently verified):\n" + listed_skills)
    missing = find_unlisted_requirements(resume, job)
    parts.append("DETERMINISTIC_NOT_ESTABLISHED_REQUIREMENTS: " + ("; ".join(missing) if missing else "None"))
    return "\n".join(parts)[:8000]

def generate_reference_answer(llm, question, profile_context=""):
    if llm is None or not getattr(llm, "ready", False):
        return None
    if is_closing_question(question):
        return None

    schema = {
        'type': 'object',
        'properties': {
            'resume_evidence': {'type': 'string'},
            'relevant_evaluation_points': {'type': 'string'},
            'not_established_by_resume': {'type': 'string'}
        },
        'required': ['resume_evidence', 'relevant_evaluation_points', 'not_established_by_resume']
    }
    result = llm.json(
        system=(
            "You are an independent interview assessor. Generate a factual evaluation rubric, never a candidate response. "
            "Use only explicit candidate facts in the resume context. Extract concrete project names, technologies, methods, and measurable achievements. "
            "For background questions, mention the most relevant actual projects and what the candidate built. "
            "Keep resume evidence separate from job expectations. A job requirement is not proof of candidate experience. "
            "Under NOT ESTABLISHED BY RESUME, list only relevant requirements absent from the resume. "
            "The resume explicitly supports cloud deployment, multilingual NLP, n8n, Groq, Hugging Face, Python, Scikit-learn, and Streamlit; do not call these unsupported. "
            "Do not invent facts or claim resume statements are independently verified. Do not repeat the full job requirements list. Be specific and concise."
        ),
        user=(
            "QUESTION:\n" + str(question)
            + "\n\nJOB AND CANDIDATE CONTEXT:\n"
            + (profile_context or "No profile context available.")
            + "\n\nWrite a neutral evaluation guide, not a sample candidate response. Use third-person or bullet-point wording, never first person. Separate what the résumé supports from what the job expects. "
            "Do not mention any employer unless the question explicitly asks for it. "
            "Do not turn résumé claims into verified interview evidence. Return JSON only."
        ),
        purpose="reference_answer",
        schema=schema,
        max_tokens=350,
        timeout=90,
        retries=1,
    )
    if not result:
        return None
    deterministic_line = next((line for line in str(profile_context).splitlines() if line.startswith("DETERMINISTIC_NOT_ESTABLISHED_REQUIREMENTS:")), "")
    deterministic_missing = deterministic_line.split(":", 1)[1].strip() if deterministic_line else ""
    sections = (
        ("RESUME EVIDENCE", result.get("resume_evidence", "")),
        ("RELEVANT EVALUATION POINTS", result.get("relevant_evaluation_points", "")),
        ("NOT ESTABLISHED BY RESUME", deterministic_missing or "None identified by the deterministic résumé evidence checker.")
    )
    answer = "\n\n".join(label + ":\n\n" + str(value).strip() for label, value in sections if str(value).strip())
    return answer or None


def generate_all(llm, questions, profile_context=""):
    references = {}
    for question in questions:
        if is_closing_question(question):
            continue
        answer = generate_reference_answer(llm, question, profile_context)
        if answer:
            references[question] = answer
    return references
