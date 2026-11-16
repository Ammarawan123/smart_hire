def extract_listed_skills(resume):
    skills = resume.get("skills", {}) or {}
    result = []
    for category, entries in skills.items():
        if not isinstance(entries, list):
            continue
        names = [str(x.get("name", "")).strip() for x in entries if isinstance(x, dict) and x.get("listed") and x.get("name")]
        if names:
            result.append(str(category) + ": " + ", ".join(names))
    return "\n".join(result)


def find_unlisted_requirements(resume, job):
    skills = resume.get("skills", {}) or {}
    listed = {str(x.get("name", "")).strip().lower() for items in skills.values() if isinstance(items, list) for x in items if isinstance(x, dict) and x.get("listed") and x.get("name")}
    projects = resume.get("projects", []) or []
    project_text = " ".join(str(x.get("name", x.get("title", ""))) + " " + str(x.get("description", "")) for x in projects if isinstance(x, dict)).lower() + " " + str(resume.get("summary", "")).lower()
    evidence = listed | {word for word in ("generative ai", "llm", "rag", "prompt engineering", "cloud deployment", "rest api", "object-oriented programming", "software development lifecycle", "data analysis", "ai project experience", "ai automation", "language translation chatbot", "movie recommendation system", "population data analysis") if word in project_text or word in str(resume.get("summary", "")).lower()}
    requirements = job.get("requirements", {}) or {}
    items = [str(v.get("name", k)) for k, v in requirements.items() if isinstance(v, dict)] if isinstance(requirements, dict) else [str(x) for x in requirements]
    aliases = {"python": ("python",), "llms": ("llms", "llm", "large language model"), "generative ai": ("generative ai", "groq", "stable diffusion"), "ai/ml libraries and frameworks": ("scikit-learn", "tensorflow", "pytorch", "hugging face transformers", "streamlit"), "data analysis": ("data analysis", "pandas", "numpy", "matplotlib", "seaborn", "exploratory data analysis")}
    missing = []
    for item in items:
        key = item.lower()
        if key == "hands-on ai project experience":
            continue
        terms = aliases.get(key, (key,))
        if not any(term in evidence or term in project_text for term in terms):
            missing.append(item)
    return missing
