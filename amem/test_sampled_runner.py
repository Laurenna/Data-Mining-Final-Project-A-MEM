"""Offline orchestration test: synthetic answers only; never calls an API."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import run_sampled_locomo as batch


def main():
    source=Path(sys.argv[1]).resolve();dataset=Path(sys.argv[2]).resolve()
    original=batch.pilot.run_pilot
    calls=[]
    def fake_run(args,plan,sample,session,stopped):
        calls.append(args.sample_index)
        existing=batch.pilot.read_results(Path(plan['run_directory'])/'answers.jsonl',plan['fingerprint'])
        for i in plan['selected_question_indices']:
            if i in existing:continue
            session.begin('offline-test')
            batch.pilot.append_jsonl(Path(plan['run_directory'])/'answers.jsonl',
               {'fingerprint':plan['fingerprint'],'qa_index':i,'sample_index':args.sample_index,
                'metrics':{'f1':0.0,'bleu1':0.0}})
        return {'completed_questions':len(plan['selected_question_indices'])}
    batch.pilot.run_pilot=fake_run
    try:
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            sys.argv=['run_sampled_locomo.py','--source-dir',str(source),'--dataset',str(dataset),
                      '--output-dir',tmp,'--max-total-calls','50','--run']
            batch.main()
            assert len(calls)==3, calls
            sys.argv[sys.argv.index('50')]='300'
            batch.main()
            rows=[json.loads(s) for s in (Path(tmp)/'answers.jsonl').read_text().splitlines()]
            assert len(rows)==200 and len({(r['sample_index'],r['qa_index']) for r in rows})==200
            count=len(calls);batch.main();assert len(calls)==count
            manifest=json.loads((Path(tmp)/'sampling_manifest.json').read_text())
            for group in manifest['groups']:
                i=group['configuration']['sample_index']
                expected=sorted(batch.random.Random(f'20261110:{i}').sample(range(group['available_questions']),20))
                assert expected==group['selected_question_indices']
        print('PASS: seeded sampling, call-cap stop, resume, 200 unique answers, completed-group reuse. No API calls.')
    finally:batch.pilot.run_pilot=original


if __name__=='__main__':main()
