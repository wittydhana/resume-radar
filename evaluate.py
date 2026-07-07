import csv
import math
from collections import Counter
from pathlib import Path
from pipeline import run_screening

BASE = Path('validation')
JD = BASE / 'jd.txt'
RES_DIR = BASE / 'resumes'
OUT_TFIDF = BASE / 'out_tfidf'
OUT_EMB = BASE / 'out_emb'
LABELS = BASE / 'labels.csv'

RELEVANCE = {'Shortlist': 2, 'Watchlist': 1, 'Reject': 0}


def load_labels(path):
    d = {}
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            d[Path(r['file']).name] = r['expected']
    return d


def score_results(df, labels):
    preds = df.set_index('file')['decision'].to_dict()
    y_true = [labels[f] for f in labels]
    y_pred = [preds.get(f, 'Reject') for f in labels]
    cm = Counter(zip(y_true, y_pred))

    total = len(labels)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / total

    metrics = {}
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for label in ['Shortlist', 'Watchlist', 'Reject']:
        tp = cm[(label, label)]
        fp = sum(cm[(other, label)] for other in ['Shortlist', 'Watchlist', 'Reject'] if other != label)
        fn = sum(cm[(label, other)] for other in ['Shortlist', 'Watchlist', 'Reject'] if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {'precision': precision, 'recall': recall, 'f1': f1}
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    macro_precision = precision_sum / 3
    macro_recall = recall_sum / 3
    macro_f1 = f1_sum / 3
    ranking = list(df['file'])
    top5 = ranking[:5]
    top5_accuracy = sum(1 for f in top5 if labels.get(f) == 'Shortlist') / min(5, len(labels))
    ndcg = compute_ndcg(ranking, labels)
    return accuracy, cm, metrics, top5_accuracy, ndcg, macro_precision, macro_recall, macro_f1


def compute_ndcg(ranking, labels):
    def dcg(scores):
        return sum((2**rel - 1) / (math.log2(idx + 2)) for idx, rel in enumerate(scores))

    relevances = [RELEVANCE[labels.get(f, 'Reject')] for f in ranking]
    ideal = sorted(relevances, reverse=True)
    actual = dcg(relevances)
    ideal_dcg = dcg(ideal)
    return actual / ideal_dcg if ideal_dcg > 0 else 0.0


def run_and_eval(use_embeddings=False, out=OUT_TFIDF):
    out.mkdir(parents=True, exist_ok=True)
    df = run_screening(JD, RES_DIR, out, use_embeddings=use_embeddings)
    labels = load_labels(LABELS)
    return df, score_results(df, labels)


def write_report(tf_results, emb_results):
    r = BASE / 'report.md'
    lines = ['# Validation Report', '']
    for label, (accuracy, cm, metrics, top5, ndcg, mp, mr, mf1) in [('TF-IDF', tf_results), ('Embeddings', emb_results)]:
        lines.append(f'## {label}')
        lines.append(f'- Accuracy: {accuracy:.2f}')
        lines.append(f'- Top-5 accuracy: {top5:.2f}')
        lines.append(f'- NDCG: {ndcg:.2f}')
        lines.append(f'- Macro precision: {mp:.2f}')
        lines.append(f'- Macro recall: {mr:.2f}')
        lines.append(f'- Macro F1: {mf1:.2f}')
        lines.append('')
        lines.append('### Confusion matrix')
        for k, v in cm.items():
            lines.append(f'- {k}: {v}')
        lines.append('')
        for cls, m in metrics.items():
            lines.append(f'- {cls} precision: {m["precision"]:.2f}, recall: {m["recall"]:.2f}, f1: {m["f1"]:.2f}')
        lines.append('')
    r.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {r}')


if __name__ == '__main__':
    df_tf, tf_results = run_and_eval(use_embeddings=False, out=OUT_TFIDF)
    df_emb, emb_results = run_and_eval(use_embeddings=True, out=OUT_EMB)
    write_report(tf_results, emb_results)
    print('Done')
