"""Offline audit of saved A-Mem pilot answers. No API or model calls.

F1 uses the repository's set-of-tokens definition (not multiset F1).
BLEU-1 uses NLTK tokenization, smoothing method1 and brevity penalty.
Evidence coverage is literal substring matching, NOT semantic retrieval recall.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def tokenize(text):
    return str(text).lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()


def score(prediction, reference):
    if not prediction or not reference:
        return {'f1': 0.0, 'bleu1': 0.0}
    prediction, reference = str(prediction).strip(), str(reference).strip()
    a, b = set(tokenize(prediction)), set(tokenize(reference))
    f1 = 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0
    bleu = sentence_bleu([nltk.word_tokenize(reference.lower())],
                         nltk.word_tokenize(prediction.lower()), weights=(1, 0, 0, 0),
                         smoothing_function=SmoothingFunction().method1)
    return {'f1': f1, 'bleu1': bleu}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--legacy-c5-reference', action='store_true',
                        help='Use adversarial_answer as category-5 reference, matching legacy evaluation')
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.answers.read_text().splitlines() if s.strip()]
    if not rows:
        raise SystemExit('No saved answers found.')
    dataset = json.loads(args.dataset.read_text())
    seen, records, evidence_rows, problems, groups = set(), [], [], [], defaultdict(list)
    for r in rows:
        key = (r['sample_index'], r['qa_index'])
        if key in seen:
            problems.append(f'Duplicate answer: {key}')
        seen.add(key)
        sample = dataset[r['sample_index']]
        q = sample['qa'][r['qa_index']]
        category = int(q['category'])
        reference = (q.get('adversarial_answer') if args.legacy_c5_reference else
                     'Not mentioned in the conversation') if category == 5 else q.get('answer')
        matching = (r['question'] == q['question'] and str(r['reference']) == str(reference)
                    and int(r['category']) == category
                    and (not r.get('sample_id') or r['sample_id'] == sample.get('sample_id')))
        if not matching:
            problems.append(f'Dataset/reference mismatch: {key}')
        computed = score(r['prediction'], reference)
        metrics_match = all(math.isclose(computed[m], r['metrics'][m], abs_tol=1e-12, rel_tol=1e-12)
                            for m in computed)
        if not metrics_match:
            problems.append(f'Metric mismatch: {key}')
        turns = {t['dia_id']: t for k, v in sample['conversation'].items()
                 if k.startswith('session_') and isinstance(v, list) for t in v}
        context = r['raw_context']
        evidence = []
        for eid in q.get('evidence', []):
            turn = turns.get(eid)
            text = turn.get('text', '') if turn else ''
            present = bool(text) and text in context
            evidence.append({'sample_index': key[0], 'qa_index': key[1], 'evidence_id': eid,
                             'source_found': turn is not None, 'text': text,
                             'literal_text_present': present})
        evidence_rows.extend(evidence)
        rec = {'sample_index': key[0], 'qa_index': key[1], 'category': category,
               'question': r['question'], 'prediction': r['prediction'], 'reference': reference,
               'saved_f1': r['metrics']['f1'], 'recomputed_f1': computed['f1'],
               'saved_bleu1': r['metrics']['bleu1'], 'recomputed_bleu1': computed['bleu1'],
               'dataset_match': matching, 'metrics_match': metrics_match,
               'context_characters': len(context), 'annotated_evidence_count': len(evidence),
               'literal_evidence_present_count': sum(e['literal_text_present'] for e in evidence),
               'manual_error_type': '', 'manual_explanation': ''}
        records.append(rec)
        groups[category].append(computed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in [('per_question.csv', records), ('evidence_checks.csv', evidence_rows)]:
        if values:
            with (args.output_dir / filename).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    summary = {'questions': len(records), 'all_checks_passed': not problems, 'problems': problems,
               'mean_metrics': {m: statistics.mean(r['recomputed_' + m] for r in records)
                                for m in ['f1', 'bleu1']},
               'by_category': {k: {'count': len(v), **{m: statistics.mean(x[m] for x in v)
                              for m in ['f1', 'bleu1']}} for k, v in groups.items()},
               'inputs': {'answers_sha256': sha(args.answers), 'dataset_sha256': sha(args.dataset),
                          'script_sha256': sha(Path(__file__))},
               'nltk_version': nltk.__version__,
               'limitations': ['Literal evidence matching is not semantic coverage or retrieval recall.',
                               'Metrics are lexical overlap, not human correctness judgments.',
                               'Character count is not token count.',
                               'Saved answer subset may not represent the full benchmark.']}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    review = ['# Saved-answer review', '', 'Automatic checks and manual analysis are separate.',
              'Literal absence does not establish semantic evidence absence.', '']
    for r in records:
        review += [f'## Sample {r["sample_index"]}, QA {r["qa_index"]} (zero-based)',
                   f'Question: {r["question"]}', f'Prediction: {r["prediction"]}',
                   f'Reference: {r["reference"]}',
                   f'F1: {r["recomputed_f1"]:.6f}; BLEU-1: {r["recomputed_bleu1"]:.6f}',
                   'Annotated evidence:']
        review += [f'- {e["evidence_id"]} (literal match={e["literal_text_present"]}): {e["text"]}'
                   for e in evidence_rows if (e['sample_index'], e['qa_index']) == (r['sample_index'], r['qa_index'])]
        review += ['Manual error classification: [to be reviewed]', 'Manual explanation: [to be reviewed]', '']
    (args.output_dir / 'review.md').write_text('\n'.join(review))
    print(json.dumps({k: v for k, v in summary.items() if k not in ['inputs', 'limitations']}, indent=2))
    print('Output:', args.output_dir.resolve())
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
