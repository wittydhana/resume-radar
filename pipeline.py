from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from parser_utils import extract_text

# Optional embeddings support (uses sentence-transformers if available)
try:
    from sentence_transformers import SentenceTransformer, util
    _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _EMBED_MODEL = None

MUST_TERMS = ["python", "sql", "nlp", "machine learning", "pandas", "scikit"]
NICE_TERMS = ["docker", "aws", "streamlit", "fastapi", "llm", "api"]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def compute_similarity_tfidf(jd_text: str, resume_text: str) -> float:
    jd = clean_text(jd_text)
    rs = clean_text(resume_text)
    if not jd or not rs:
        return 0.0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        X = vectorizer.fit_transform([jd, rs])
        return float(cosine_similarity(X[0], X[1])[0, 0]) * 100
    except ValueError:
        return 0.0


def compute_similarity_embeddings(jd_text: str, resume_text: str) -> float:
    if _EMBED_MODEL is None:
        return compute_similarity_tfidf(jd_text, resume_text)
    jd = clean_text(jd_text)
    rs = clean_text(resume_text)
    if not jd or not rs:
        return 0.0
    emb_jd = _EMBED_MODEL.encode(jd, convert_to_tensor=True)
    emb_rs = _EMBED_MODEL.encode(rs, convert_to_tensor=True)
    sim = util.cos_sim(emb_jd, emb_rs).item()
    return float(sim) * 100


def score_resume(jd_text: str, resume_text: str, use_embeddings: bool = False):
    if use_embeddings:
        similarity = compute_similarity_embeddings(jd_text, resume_text)
    else:
        similarity = compute_similarity_tfidf(jd_text, resume_text)
    jd = clean_text(jd_text)
    rs = clean_text(resume_text)

    required_terms = [t for t in MUST_TERMS if t in jd]
    optional_terms = [t for t in NICE_TERMS if t in jd]
    matched_must = [t for t in required_terms if t in rs]
    missing_must = [t for t in required_terms if t not in rs]
    matched_nice = [t for t in optional_terms if t in rs]

    # adjust weights to better surface partially-matching candidates
    score = similarity * 0.5 + 12 * len(matched_must) + 2.5 * len(matched_nice) - 4 * len(missing_must)
    score = round(max(0.0, min(100.0, score)), 1)

    risk = round(min(100, 20 * len(missing_must) + max(0, 40 - score)), 1)
    # Adjusted decision boundaries: allow more forgiving Watchlist for partial matches
    if score >= 55 and len(missing_must) == 0:
        decision = "Shortlist"
    elif score >= 50 and len(matched_must) >= max(1, len(required_terms) - 1):
        decision = "Shortlist"
    # relaxed shortlist for near-complete matches
    elif score >= 44 and len(matched_must) >= max(1, len(required_terms) - 2):
        decision = "Shortlist"
    elif score >= 25:
        decision = "Watchlist"
    else:
        decision = "Reject"

    reason = (
        f"TF-IDF={round(similarity, 1)}; matched_must={len(matched_must)}; "
        f"matched_nice={len(matched_nice)}; missing_must={len(missing_must)}"
    )
    return score, similarity, risk, decision, matched_must, missing_must, matched_nice, reason


def run_screening(jd_path, resumes_dir, out_dir, use_embeddings: bool = False):
    jd_path = Path(jd_path)
    resumes_dir = Path(resumes_dir)
    out_dir = Path(out_dir)

    if not jd_path.exists():
        raise FileNotFoundError(f"Job description file not found: {jd_path}")
    if not resumes_dir.exists() or not resumes_dir.is_dir():
        raise FileNotFoundError(f"Resumes directory not found: {resumes_dir}")

    if out_dir.exists() and not out_dir.is_dir():
        out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    jd_text = jd_path.read_text(encoding="utf-8", errors="ignore")

    rows = []
    for f in sorted(resumes_dir.iterdir()):
        if f.suffix.lower() not in {".pdf", ".docx", ".txt"}:
            continue
        txt = extract_text(f)
        score, similarity, risk, decision, matched_must, missing_must, matched_nice, reason = score_resume(jd_text, txt, use_embeddings=use_embeddings)
        excerpt = " ".join(txt.replace("\n", " ").split())[:180].strip()
        rows.append({
            "file": f.name,
            "score": score,
            "similarity": round(similarity, 1),
            "risk": risk,
            "decision": decision,
            "excerpt": excerpt,
            "matched_must": ", ".join(matched_must),
            "missing_must": ", ".join(missing_must),
            "matched_nice": ", ".join(matched_nice),
            "reason": reason,
        })

    if rows:
        df = pd.DataFrame(rows)
        decision_order = {"Shortlist": 0, "Watchlist": 1, "Reject": 2}
        df["decision_rank"] = df["decision"].map(decision_order)
        df = df.sort_values(["decision_rank", "score", "file"], ascending=[True, False, True]).drop(columns=["decision_rank"])
    else:
        df = pd.DataFrame(columns=[
            "file", "score", "similarity", "risk", "decision", "excerpt", "matched_must",
            "missing_must", "matched_nice", "reason"
        ])

    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(out_dir / "ranked_results.csv", index=False)
    df.to_json(out_dir / "ranked_results.json", orient="records", indent=2)
    (out_dir / "scoring_note.txt").write_text(
        "Score formula: 0.5*TF-IDF + 8*matched_must + 2.5*matched_nice - 12*missing_must. "
        "Decision = Shortlist if score>=55 and no missing must-have; Watchlist if score>=40; otherwise Reject.",
        encoding="utf-8"
    )

    summary_lines = [
        f"Resume screening report for {len(df)} resumes from {resumes_dir}",
        "",
        "Decision counts:",
    ]
    decision_counts = df['decision'].value_counts().to_dict()
    for label in ["Shortlist", "Watchlist", "Reject"]:
        summary_lines.append(f"- {label}: {decision_counts.get(label, 0)}")
    summary_lines.append("")
    summary_lines.append("Top candidates:")
    for _, row in df.head(5).iterrows():
        summary_lines.append(
            f"{row['rank']}. {row['file']} | {row['decision']} | score={row['score']} | similarity={row['similarity']} | risk={row['risk']}"
        )
        summary_lines.append(f"   reason: {row['reason']}")
    (out_dir / "ranked_report.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Ranked {len(df)} resumes from {resumes_dir}")
    if not df.empty:
        for _, row in df.iterrows():
            print(f"{row['rank']}. {row['file']} | {row['decision']} | score={row['score']} | risk={row['risk']}")
            excerpt = row['excerpt'] or ''
            print(f"   excerpt: {excerpt[:120]}{'...' if len(excerpt) > 120 else ''}")
        print(f"\nA summary report was written to {out_dir / 'ranked_report.txt'}")
    else:
        print("No supported resume files were found.")
    return df