"""Paired full-history control for saved A-Mem QA. Default: offline preflight only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
SCHEMA = {"type": "json_schema", "json_schema": {"name": "response", "schema": {
    "type": "object", "properties": {"answer": {"type": "string"}},
    "required": ["answer"], "additionalProperties": False}, "strict": True}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def atomic_json(path, obj):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def full_history(sample):
    """Preserve source session order, all turns, dates and upstream image captions."""
    lines, count = [], 0
    conv = sample['conversation']
    for key, turns in conv.items():
        if not (key.startswith('session_') and isinstance(turns, list)):
            continue
        date = conv.get(key + '_date_time')
        if not date:
            raise ValueError('Missing session date: ' + key)
        lines.append('DATE OF CONVERSATION: ' + date)
        for turn in turns:
            text = turn.get('text', '')
            if 'img_url' in turn and 'blip_caption' in turn:
                caption = f"[Image: {turn['blip_caption']}]"
                text = caption + (' ' + text if text else '')
            lines.append(f"{turn['speaker']}: {text}")
            count += 1
        lines.append('')
    return '\n'.join(lines), count


def prepare(dataset, amem_dir):
    data = json.loads(dataset.read_text())
    manifest = json.loads((amem_dir / 'sampling_manifest.json').read_text())
    if manifest['dataset_sha256'] != sha(dataset):
        raise ValueError('Dataset differs from the A-Mem experiment.')
    rows = read_rows(amem_dir / 'answers.jsonl')
    keyed = {(r['sample_index'], r['qa_index']): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError('Duplicate A-Mem answers.')
    plans, stats, expected = [], [], set()
    for group in manifest['groups']:
        config = group['configuration']
        i = config['sample_index']
        if config['model'] != 'deepseek-flash' or config['thinking'] != 'disabled':
            raise ValueError('Unexpected source model or thinking mode.')
        if config['legacy_c5_reference']:
            raise ValueError('This control requires the corrected category-5 reference.')
        if digest(data[i]) != config['sample_sha256']:
            raise ValueError('Conversation hash mismatch.')
        context, turns = full_history(data[i])
        if turns != group['complete_history_turns']:
            raise ValueError('History turn count mismatch.')
        stats.append({'sample_index': i, 'turns': turns, 'history_characters': len(context),
                      'history_utf8_bytes': len(context.encode()), 'questions': len(group['selected_question_indices'])})
        for q in group['selected_question_indices']:
            key = (i, q)
            if key in expected:
                raise ValueError('Duplicate selected question.')
            expected.add(key)
            r = keyed[key]
            qa = data[i]['qa'][q]
            reference = 'Not mentioned in the conversation' if int(qa['category']) == 5 else qa.get('answer')
            if (r['fingerprint'] != group['fingerprint'] or r['question'] != qa['question']
                    or r['reference'] != reference or int(r['category']) != int(qa['category'])):
                raise ValueError('A-Mem answer metadata mismatch.')
            old = r['raw_context']
            if not old or r['prompt'].count(old) != 1:
                raise ValueError('Cannot uniquely replace the saved retrieval context.')
            prompt = r['prompt'].replace(old, context, 1)
            # Byte count is a deliberately conservative sizing proxy, NOT exact tokenization.
            # Includes ample room for schema/system formatting and maximum retry output.
            if len(prompt.encode()) + 8192 > 1_000_000:
                raise ValueError('Prompt exceeds conservative context sizing guard; no truncation allowed.')
            plans.append({'sample_index': i, 'sample_id': data[i].get('sample_id'), 'qa_index': q,
                          'question': r['question'], 'category': r['category'], 'reference': reference,
                          'prompt': prompt, 'raw_context': context,
                          'temperature': config['temperature_c5'] if int(r['category']) == 5 else config['temperature'],
                          'max_output_tokens': config['max_output_tokens']})
    if expected != set(keyed):
        raise ValueError('Saved A-Mem answers do not exactly match selected questions.')
    if len({p['max_output_tokens'] for p in plans}) != 1:
        raise ValueError('Mixed output limits are not supported.')
    identity = {'method': 'full_history_control_v1', 'model': 'deepseek-flash', 'thinking': 'disabled',
                'dataset_sha256': sha(dataset), 'amem_answers_sha256': sha(amem_dir / 'answers.jsonl'),
                'sampling_manifest_sha256': sha(amem_dir / 'sampling_manifest.json'),
                'prompt_sha256': digest([p['prompt'] for p in plans]),
                'settings_sha256': digest([(p['temperature'], p['max_output_tokens']) for p in plans]),
                'code_sha256': {n: sha(ROOT / n) for n in ('run_full_history.py', 'deepseek_adapter.py', 'audit_evaluation.py')}}
    return plans, {'fingerprint': digest(identity), 'configuration': identity, 'groups': stats,
                   'questions': len(plans), 'history_policy': 'All turns retained; no truncation or summarization',
                   'prompt_policy': 'Replace only raw_context in each saved A-Mem answer prompt; preserve all other text',
                   'context_limit_source': 'https://api-docs.deepseek.com/quick_start/pricing/',
                   'context_limit_checked_date': '2026-11-12',
                   'caveat': 'Paired full-history control, not an exact reproduction of the original LoCoMo baseline implementation.'}


def load_existing(output, fingerprint, plans):
    path = output / 'answers.jsonl'
    rows = read_rows(path) if path.exists() else []
    allowed = {(p['sample_index'], p['qa_index']): p for p in plans}
    seen = set()
    for r in rows:
        key = (r['sample_index'], r['qa_index'])
        if key in seen or key not in allowed or r['fingerprint'] != fingerprint:
            raise ValueError('Saved answers are incompatible or duplicated.')
        if any(r[k] != allowed[key][k] for k in ('question', 'reference', 'prompt', 'raw_context')):
            raise ValueError('Saved answer inputs differ.')
        seen.add(key)
    return rows, seen


def execute(plans, manifest, output, max_calls, max_new_questions=None):
    from audit_evaluation import score
    import nltk
    import deepseek_adapter as adapter
    nltk.word_tokenize('Tokenizer readiness check')  # Fail before any paid request.
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Settings/code changed; use a different output directory.')
    rows, seen = load_existing(output, manifest['fingerprint'], plans)
    atomic_json(manifest_path, manifest)
    session = adapter.configure_session(output / 'usage.jsonl', max_calls, plans[0]['max_output_tokens'])
    session.phase = 'full_history_qa'
    status, reason, new_count = 'running', '', 0
    try:
        controller = None
        for plan in plans:
            key = (plan['sample_index'], plan['qa_index'])
            if key in seen:
                continue
            if max_new_questions is not None and new_count >= max_new_questions:
                status, reason = 'paused', 'Requested new-question limit reached'
                break
            if controller is None:
                controller = adapter.DeepSeekController(manifest['configuration']['model'])
            started = time.perf_counter()
            prediction = json.loads(controller.get_completion(plan['prompt'], SCHEMA, plan['temperature']))['answer']
            row = dict(plan, fingerprint=manifest['fingerprint'], prediction=prediction,
                       metrics=score(prediction, plan['reference']), elapsed_seconds=time.perf_counter() - started)
            with (output / 'answers.jsonl').open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            rows.append(row)
            seen.add(key)
            new_count += 1
            group_done = sum(r['sample_index'] == key[0] for r in rows)
            group_total = sum(p['sample_index'] == key[0] for p in plans)
            print(f"Saved {len(rows)}/{len(plans)} QA | conversation {key[0]+1} | {group_done}/{group_total} | {row['elapsed_seconds']:.1f}s", flush=True)
        else:
            status = 'completed'
    except adapter.RunStopped as exc:
        status, reason = 'stopped', str(exc)
    except KeyboardInterrupt:
        status, reason = 'stopped', 'Interrupted by user; completed answers are saved'
    except Exception as exc:
        status, reason = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        summary = {'status': status, 'reason': reason, 'completed_questions': len(rows), 'total_questions': len(plans),
                   'mean_metrics': {k: sum(r['metrics'][k] for r in rows)/len(rows) for k in ('f1', 'bleu1')} if rows else {},
                   'usage': session.summary()}
        atomic_json(output / 'summary.json', summary)
        print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--amem-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-calls', type=int, default=300, help='Cumulative request cap, including retries')
    parser.add_argument('--max-new-questions', type=int, help='Optional smoke-run limit; omit to finish all questions')
    parser.add_argument('--run', action='store_true', help='Enable paid API calls')
    args = parser.parse_args()
    if args.max_calls < 1 or (args.max_new_questions is not None and args.max_new_questions < 1):
        parser.error('Limits must be positive.')
    plans, manifest = prepare(args.dataset, args.amem_dir)
    rows, _ = load_existing(args.output_dir, manifest['fingerprint'], plans)
    for g in manifest['groups']:
        print(f"Conversation {g['sample_index']+1}: {g['turns']} full-history turns; {g['questions']} selected QA; {g['history_characters']} history characters")
    print(f"Questions: {len(plans)}; already saved: {len(rows)}; remaining: {len(plans)-len(rows)}")
    print('One API request per remaining question, excluding bounded retries. No memory building.')
    if args.run:
        execute(plans, manifest, args.output_dir, args.max_calls, args.max_new_questions)
    else:
        print('OFFLINE PLAN ONLY: no API calls, downloads or output changes.')


if __name__ == '__main__':
    main()
