import pandas as pd
from pathlib import Path

out = Path("output")
csv = out / "ranked_results.csv"
md = out / "presentation.md"

if not csv.exists():
    print("No ranked_results.csv found. Run python main.py first.")
else:
    df = pd.read_csv(csv)
    lines = ["# Resume Radar — Top Candidates\n"]
    for _, r in df.head(10).iterrows():
        lines.append(f"## {r['rank']}. {r['file']} — {r['decision']}\n")
        lines.append(f"- Score: {r['score']}  \n")
        lines.append(f"- Similarity: {r.get('similarity', '')}  \n")
        lines.append(f"- Risk: {r['risk']}  \n")
        excerpt = (r['excerpt'] or '')
        lines.append(f"- Excerpt: {excerpt[:300]}{'...' if len(excerpt)>300 else ''}\n")
        lines.append(f"- Matched must-haves: {r.get('matched_must','')}\n")
        lines.append(f"- Missing must-haves: {r.get('missing_must','')}\n")
        lines.append(f"- Reason: {r.get('reason','')}\n")
        lines.append('\n')
    md.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote {md}")
