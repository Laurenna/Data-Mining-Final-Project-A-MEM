"""Compare paired saved answers without API calls. Positive differences favor full history."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict


def load(path):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    keyed = {(r['sample_index'], r['qa_index']): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError('Duplicate question keys')
    return keyed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--amem', type=Path, required=True)
    p.add_argument('--full-history', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    am, full = load(a.amem), load(a.full_history)
    if not am or am.keys() != full.keys():
        raise ValueError('Question sets differ or are empty')
    paired, groups = [], defaultdict(list)
    for key in sorted(am):
        x, y = am[key], full[key]
        for field in ('question', 'reference', 'category', 'sample_id'):
            if x.get(field) != y.get(field):
                raise ValueError(f'Mismatched {field}: {key}')
        if (not x['raw_context'] or not y['raw_context']
                or x['prompt'].count(x['raw_context']) != 1 or y['prompt'].count(y['raw_context']) != 1
                or x['prompt'].replace(x['raw_context'], '<CONTEXT>', 1)
                != y['prompt'].replace(y['raw_context'], '<CONTEXT>', 1)):
            raise ValueError(f'Prompt instructions differ: {key}')
        row = dict(sample_index=key[0], qa_index=key[1], category=x['category'], question=x['question'],
                   reference=x['reference'], amem_prediction=x['prediction'], full_history_prediction=y['prediction'])
        for m in ('f1', 'bleu1'):
            row['amem_'+m] = x['metrics'][m]
            row['full_history_'+m] = y['metrics'][m]
            row['difference_'+m] = y['metrics'][m]-x['metrics'][m]
        paired.append(row)
        for group in ('overall', f"category_{x['category']}", f'conversation_{key[0]+1}'):
            groups[group].append(row)
    summary = []
    for name, rows in groups.items():
        entry = {'group': name, 'questions': len(rows)}
        for m in ('f1', 'bleu1'):
            for prefix in ('amem_', 'full_history_', 'difference_'):
                entry[prefix+m] = sum(r[prefix+m] for r in rows)/len(rows)
        summary.append(entry)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [('paired_questions.csv', paired), ('comparison.csv', summary)]:
        with (a.output_dir/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    report = {'questions': len(paired), 'exact_question_and_prompt_pairing_verified': True,
              'difference_definition': 'full_history minus A-Mem', 'results': summary,
              'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.amem, a.full_history)},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': 'Descriptive sampled comparison, not statistical significance or full paper replication. Run audit_evaluation.py on each input to independently verify saved scores.'}
    (a.output_dir/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary[0], indent=2))


if __name__ == '__main__':
    main()
