from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from parser_utils import extract_text

_EMBED_MODEL = None
_EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'


def load_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    except Exception:
        _EMBED_MODEL = None
    return _EMBED_MODEL

SYNONYM_MAP = {
    'py': 'python',
    'js': 'javascript',
    'javascript': 'javascript',
    'typescript': 'typescript',
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'aws': 'aws',
    'docker': 'docker',
    'k8s': 'kubernetes',
    'kubernetes': 'kubernetes',
    'fastapi': 'fastapi',
    'flask': 'flask',
    'django': 'django',
    'spark': 'spark',
    'sql': 'sql',
    'postgres': 'postgresql',
    'postgresql': 'postgresql',
    'mysql': 'mysql',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit': 'scikit-learn',
    'scikit-learn': 'scikit-learn',
    'sklearn': 'scikit-learn',
    'tensorflow': 'tensorflow',
    'pytorch': 'pytorch',
    'transformers': 'transformers',
    'llm': 'llm',
    'spring boot': 'spring boot',
    'kafka': 'kafka',
    'react': 'react',
    'flutter': 'flutter',
    'java': 'java',
    'c++': 'c++',
    'c#': 'c#',
    'golang': 'golang',
    'go': 'go',
}

TERM_ALIASES = {
    'machine learning': {'ml', 'artificial intelligence', 'ai', 'data science', 'applied scientist', 'text analytics', 'transformers', 'modeling', 'classification', 'deep learning'},
    'nlp': {'natural language processing', 'language model', 'text processing', 'text analytics', 'named entity recognition', 'question answering', 'transformers'},
    'sql': {'sql', 'database', 'postgres', 'postgresql', 'mysql'},
    'pandas': {'pandas', 'dataframe', 'data frames', 'data processing'},
    'scikit-learn': {'scikit-learn', 'sklearn', 'scikit', 'transformers', 'tensorflow', 'pytorch', 'deep learning'},
    'python': {'python'},
}

COMMON_STOPWORDS = {
    'and', 'or', 'with', 'the', 'a', 'an', 'of', 'to', 'for', 'in', 'on', 'at', 'by', 'its',
    'experience', 'candidate', 'role', 'team', 'work', 'years', 'year', 'strong', 'knowledge',
    'ability', 'using', 'including', 'preferred', 'required', 'must', 'needs', 'will',
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_term(term: str) -> str:
    term = term.strip().lower()
    term = re.sub(r"[\(\)\[\]:\"]", "", term)
    term = re.sub(r"\s+", " ", term)
    if term in SYNONYM_MAP:
        return SYNONYM_MAP[term]
    return term


def extract_phrases(text: str) -> list[str]:
    text = text.replace('•', '\n').replace('·', '\n')
    pieces = re.split(r'[\n;•·]+', text)
    phrases = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidates = re.split(r',|/|\band\b|\bor\b', piece)
        for candidate in candidates:
            phrase = candidate.strip()
            if not phrase:
                continue
            words = [w for w in re.findall(r"[a-z0-9#+\-.]+", phrase.lower())]
            if len(words) == 0 or len(words) > 5:
                continue
            if len(words) == 1 and words[0] in COMMON_STOPWORDS:
                continue
            phrases.append(' '.join(words))
    return phrases


def phrase_is_skill(phrase: str) -> bool:
    if not phrase:
        return False
    words = phrase.split()
    if len(words) > 5:
        return False
    if any(word in SYNONYM_MAP for word in words):
        return True
    if any(char.isdigit() for char in phrase):
        return True
    if len(words) >= 2:
        return True
    return len(phrase) <= 4


def extract_requirements(jd_text: str) -> tuple[list[str], list[str]]:
    text = clean_text(jd_text)
    lines = text.split('\n')
    required = []
    optional = []
    active = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r'\b(required|must have|must|must include|needs|required skills|you will have|minimum)\b', stripped):
            active = 'required'
            continue
        if re.search(r'\b(preferred|nice to have|optional|desirable|wish list|bonus)\b', stripped):
            active = 'optional'
            continue
        candidates = extract_phrases(stripped)
        for phrase in candidates:
            if not phrase_is_skill(phrase):
                continue
            phrase = normalize_term(phrase)
            if active == 'required':
                required.append(phrase)
            elif active == 'optional':
                optional.append(phrase)
            else:
                if re.search(r'\b(skill|experience|technology|tool|framework|language|platform|cloud|data|ai|ml)\b', phrase) or phrase in SYNONYM_MAP:
                    required.append(phrase)
                else:
                    optional.append(phrase)
    required = [normalize_term(p) for p in dict.fromkeys(required) if p and p not in COMMON_STOPWORDS]
    optional = [normalize_term(p) for p in dict.fromkeys(optional) if p and p not in COMMON_STOPWORDS and p not in required]
    if not required:
        all_phrases = [normalize_term(p) for p in extract_phrases(text) if phrase_is_skill(p)]
        required = all_phrases[:5]
    return required, optional


def compute_similarity_tfidf(jd_text: str, resume_text: str) -> float:
    jd = clean_text(jd_text)
    rs = clean_text(resume_text)
    if not jd or not rs:
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    try:
        X = vectorizer.fit_transform([jd, rs])
        return float(cosine_similarity(X[0], X[1])[0, 0]) * 100
    except ValueError:
        return 0.0


def compute_similarity_embeddings(jd_text: str, resume_text: str) -> float:
    model = load_embed_model()
    if model is None:
        return compute_similarity_tfidf(jd_text, resume_text)
    try:
        from sentence_transformers.util import cos_sim
    except Exception:
        return compute_similarity_tfidf(jd_text, resume_text)

    jd = clean_text(jd_text)
    rs = clean_text(resume_text)
    if not jd or not rs:
        return 0.0
    emb_jd = model.encode(jd, convert_to_tensor=True)
    emb_rs = model.encode(rs, convert_to_tensor=True)
    sim = cos_sim(emb_jd, emb_rs).item()
    return float(sim) * 100


def match_skills(terms: list[str], text: str) -> list[str]:
    normalized_text = clean_text(text)
    matched = []
    for term in terms:
        normalized = normalize_term(term)
        found = False
        if normalized in normalized_text:
            found = True
        elif ' ' in normalized:
            if all(part in normalized_text for part in normalized.split()):
                found = True
        else:
            if re.search(rf'\b{re.escape(normalized)}\b', normalized_text):
                found = True

        if not found and normalized in TERM_ALIASES:
            aliases = TERM_ALIASES[normalized]
            for alias in aliases:
                if alias in normalized_text:
                    found = True
                    break

        if found:
            matched.append(normalized)
    return matched


def extract_experience_years(text: str) -> int:
    matches = re.findall(r'(\d+)\+?\s*(?:years|yrs|y)\b', text.lower())
    years = [int(m) for m in matches if m.isdigit()]
    return max(years) if years else 0


def build_explanation(resume_name: str, required: list[str], optional: list[str], matched_required: list[str], matched_optional: list[str], missing_required: list[str], similarity: float, score: float, experience: int) -> str:
    lines = [f'Candidate: {resume_name}', f'Score: {score}', f'Semantic similarity: {round(similarity, 1)}%']
    if required:
        lines.append(f"Required skills found: {', '.join(matched_required) if matched_required else 'None'}")
        lines.append(f"Missing required skills: {', '.join(missing_required) if missing_required else 'None'}")
    if optional:
        lines.append(f"Optional skills found: {', '.join(matched_optional) if matched_optional else 'None'}")
    if experience:
        lines.append(f"Extracted experience: {experience} years")
    if missing_required:
        lines.append("Recommendation: review missing requirements before shortlisting.")
    else:
        lines.append("Recommendation: strong match for the role.")
    lines.append(f"Required coverage: {len(matched_required)}/{len(required)}")
    lines.append(f"Optional matches: {len(matched_optional)}")
    return " | ".join(lines)


def score_resume(jd_text: str, resume_text: str, use_embeddings: bool = False, resume_name: str = ''):
    required_terms, optional_terms = extract_requirements(jd_text)
    similarity = compute_similarity_embeddings(jd_text, resume_text) if use_embeddings or _EMBED_MODEL else compute_similarity_tfidf(jd_text, resume_text)
    matched_required = match_skills(required_terms, resume_text)
    missing_required = [t for t in required_terms if t not in matched_required]
    matched_optional = match_skills(optional_terms, resume_text)
    experience_years = extract_experience_years(resume_text)

    coverage = len(matched_required) / max(len(required_terms), 1)
    experience_bonus = min(experience_years, 5) * 2
    score = similarity * 0.45 + coverage * 30 + len(matched_optional) * 5 + experience_bonus - len(missing_required) * 4
    score = round(max(0.0, min(100.0, score)), 1)

    risk = round(min(100, max(0, 40 - coverage * 40) + max(0, 20 - experience_bonus)), 1)
    if score >= 40 and not missing_required:
        decision = 'Shortlist'
    elif score >= 20:
        decision = 'Watchlist'
    else:
        decision = 'Reject'

    reason = build_explanation(resume_name, required_terms, optional_terms, matched_required, matched_optional, missing_required, similarity, score, experience_years)
    return score, similarity, risk, decision, matched_required, missing_required, matched_optional, reason


def run_screening(jd_path, resumes_dir, out_dir, use_embeddings: bool = False):
    jd_path = Path(jd_path)
    resumes_dir = Path(resumes_dir)
    out_dir = Path(out_dir)

    if not jd_path.exists():
        raise FileNotFoundError(f'Resume description file not found: {jd_path}')
    if not resumes_dir.exists() or not resumes_dir.is_dir():
        raise FileNotFoundError(f'Resumes directory not found: {resumes_dir}')

    if out_dir.exists() and not out_dir.is_dir():
        out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    jd_text = jd_path.read_text(encoding='utf-8', errors='ignore')
    required_terms, optional_terms = extract_requirements(jd_text)

    rows = []
    for f in sorted(resumes_dir.iterdir()):
        if f.suffix.lower() not in {'.pdf', '.docx', '.txt'}:
            continue
        txt = extract_text(f)
        score, similarity, risk, decision, matched_required, missing_required, matched_optional, reason = score_resume(jd_text, txt, use_embeddings=use_embeddings, resume_name=f.name)
        excerpt = ' '.join(txt.replace('\n', ' ').split())[:180].strip()
        rows.append({
            'file': f.name,
            'score': score,
            'similarity': round(similarity, 1),
            'risk': risk,
            'decision': decision,
            'excerpt': excerpt,
            'matched_required': ', '.join(matched_required),
            'missing_required': ', '.join(missing_required),
            'matched_optional': ', '.join(matched_optional),
            'reason': reason,
        })

    if rows:
        df = pd.DataFrame(rows)
        decision_order = {'Shortlist': 0, 'Watchlist': 1, 'Reject': 2}
        df['decision_rank'] = df['decision'].map(decision_order)
        df = df.sort_values(['decision_rank', 'score', 'file'], ascending=[True, False, True]).drop(columns=['decision_rank'])
    else:
        df = pd.DataFrame(columns=[
            'file', 'score', 'similarity', 'risk', 'decision', 'excerpt', 'matched_required',
            'missing_required', 'matched_optional', 'reason'
        ])

    df.insert(0, 'rank', range(1, len(df) + 1))
    df.to_csv(out_dir / 'ranked_results.csv', index=False)
    df.to_json(out_dir / 'ranked_results.json', orient='records', indent=2)
    scoring_details = (
        'Score formula: 0.45*semantic_similarity + 30*required_coverage + 5*optional_matches + 2*experience_years - 4*missing_required. '
        'Decision = Shortlist if score>=40 and no missing required; Watchlist if score>=20; otherwise Reject.'
    )
    (out_dir / 'scoring_note.txt').write_text(scoring_details, encoding='utf-8')

    summary_lines = [
        f'Resume screening report for {len(df)} resumes from {resumes_dir}',
        '',
        f'Required skills extracted: {", ".join(required_terms) or "none"}',
        f'Optional skills extracted: {", ".join(optional_terms) or "none"}',
        '',
        'Decision counts:',
    ]
    decision_counts = df['decision'].value_counts().to_dict()
    for label in ['Shortlist', 'Watchlist', 'Reject']:
        summary_lines.append(f'- {label}: {decision_counts.get(label, 0)}')
    summary_lines.append('')
    summary_lines.append('Top candidates:')
    for _, row in df.head(5).iterrows():
        summary_lines.append(
            f"{row['rank']}. {row['file']} | {row['decision']} | score={row['score']} | similarity={row['similarity']} | risk={row['risk']}"
        )
        summary_lines.append(f"   reason: {row['reason']}")
    (out_dir / 'ranked_report.txt').write_text('\n'.join(summary_lines), encoding='utf-8')

    print(f'Ranked {len(df)} resumes from {resumes_dir}')
    if not df.empty:
        for _, row in df.iterrows():
            print(f"{row['rank']}. {row['file']} | {row['decision']} | score={row['score']} | risk={row['risk']}")
            excerpt = row['excerpt'] or ''
            print(f"   excerpt: {excerpt[:120]}{'...' if len(excerpt) > 120 else ''}")
        print(f"\nA summary report was written to {out_dir / 'ranked_report.txt'}")
    else:
        print('No supported resume files were found.')
    return df
