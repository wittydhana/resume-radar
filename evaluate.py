import csv
from pathlib import Path
from pipeline import run_screening
import pandas as pd

BASE = Path("validation")
JD = BASE / "jd.txt"
RES_DIR = BASE / "resumes"
OUT_TFIDF = BASE / "out_tfidf"
OUT_EMB = BASE / "out_emb"
LABELS = BASE / "labels.csv"


def load_labels(path):
    d = {}
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # normalize to basename so paths like 'resumes/foo.txt' match pipeline output 'foo.txt'
            key = Path(r['file']).name
            d[key] = r['expected']
    return d


def score_results(df, labels):
    preds = df.set_index('file')['decision'].to_dict()
    total = len(labels)
    correct = sum(1 for f,v in labels.items() if preds.get(f)==v)
    acc = correct/total if total else 0.0
    # confusion
    from collections import Counter
    pairs = [(labels[f], preds.get(f, 'Missing')) for f in labels]
    conf = Counter(pairs)
    return acc, conf


def run_and_eval(use_embeddings=False, out=OUT_TFIDF):
    out.mkdir(parents=True, exist_ok=True)
    df = run_screening(JD, RES_DIR, out, use_embeddings=use_embeddings)
    labels = load_labels(LABELS)
    acc, conf = score_results(df, labels)
    return df, acc, conf


def write_report(tf_acc, tf_conf, emb_acc, emb_conf):
    r = BASE / 'report.md'
    lines = ["# Validation Report\n"]
    lines.append(f"TF-IDF accuracy: {tf_acc:.2f}\n")
    lines.append("TF-IDF confusion (expected,predicted):\n")
    for k,v in tf_conf.items():
        lines.append(f"- {k}: {v}\n")
    lines.append('\n')
    lines.append(f"Embeddings accuracy: {emb_acc:.2f}\n")
    lines.append("Embeddings confusion (expected,predicted):\n")
    for k,v in emb_conf.items():
        lines.append(f"- {k}: {v}\n")
    r.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote {r}")


if __name__ == '__main__':
    df_tf, tf_acc, tf_conf = run_and_eval(use_embeddings=False, out=OUT_TFIDF)
    df_emb, emb_acc, emb_conf = run_and_eval(use_embeddings=True, out=OUT_EMB)
    write_report(tf_acc, tf_conf, emb_acc, emb_conf)
    print('Done')
