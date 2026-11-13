"""Full-history A-Mem evaluation with uniform seeded QA subsampling.
Default mode is an offline plan. --run enables paid calls.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import random
import run_locomo_pilot as pilot

ROOT=Path(__file__).resolve().parent
CORE=('memory_layer.py','test_advanced.py','load_dataset.py','utils.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reuse-first',type=Path)
    p.add_argument('--questions-per-group',type=int,default=20)
    p.add_argument('--selection-seed',type=int,default=20261110)
    p.add_argument('--max-total-calls',type=int,default=1000,
                   help='Cumulative new requests across this batch, excluding the reused experiment')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.questions_per_group<1 or a.max_total_calls<1:p.error('Counts must be positive')
    source=a.source_dir.expanduser().resolve(); data=json.loads(a.dataset.read_text())
    pilot.SOURCE_ROOT=source
    sys.path.insert(0,str(source))
    plans=[]
    for index,sample in enumerate(data):
        args=SimpleNamespace(dataset=str(a.dataset.resolve()),sample_index=index,questions=a.questions_per_group,
              model='deepseek-flash',max_output_tokens=1500,retrieve_k=10,temperature_c5=0.5,
              seed=42,refresh_index=False,legacy_c5_reference=False,max_calls=a.max_total_calls)
        plan,_=pilot.make_plan(args)
        selected=sorted(random.Random(f'{a.selection_seed}:{index}').sample(range(len(sample['qa'])),min(a.questions_per_group,len(sample['qa']))))
        plan['selected_question_indices']=selected
        plan['selection_method']='Uniform sample without replacement, independently per conversation'
        plan['selection_seed']=a.selection_seed
        plan['run_directory']=str(a.output_dir.resolve()/f'group_{index:02d}')
        identity={'configuration':plan['configuration'],'selection':selected,'selection_seed':a.selection_seed,
                  'core_hashes':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in CORE}}
        if index==0 and a.reuse_first:
            old=json.loads((a.reuse_first/'manifest.json').read_text())
            for k,v in old['configuration'].items():
                if k!='source_sha256' and plan['configuration'].get(k)!=v:p.error('Reuse mismatch: '+k)
            for n in CORE:
                if old['configuration']['source_sha256'][n]!=identity['core_hashes'][n]:p.error('Source changed: '+n)
            plan['fingerprint']=old['fingerprint']
            plan['configuration']=old['configuration']
            cached=pilot.read_results(a.reuse_first/'answers.jsonl',old['fingerprint'])
            if not all(i in cached for i in selected):p.error('Reused experiment lacks selected answers')
            if not (a.reuse_first/'memory_checkpoint.pkl').exists():p.error('Reuse checkpoint missing')
            plan['reuse_source']=str(a.reuse_first.resolve())
        else:plan['fingerprint']=pilot.digest(identity)
        plan['batch_identity']=pilot.digest(identity)
        target=Path(plan['run_directory'])
        if (target/'manifest.json').exists():
            previous=json.loads((target/'manifest.json').read_text())
            if previous.get('batch_identity')!=plan['batch_identity']:p.error('Existing batch settings differ; use a new output directory')
        plans.append((args,plan,sample))
    used=0
    for _,plan,_ in plans:
        ledger=Path(plan['run_directory'])/'usage.jsonl'
        if ledger.exists():used+=sum(json.loads(s)['event']=='attempt' for s in ledger.read_text().splitlines() if s.strip())
    for _,plan,_ in plans:
        print(f"Group {plan['configuration']['sample_index']}: {plan['complete_history_turns']} turns; {len(plan['selected_question_indices'])} sampled QA; "
              +('reuse existing memory and answers' if plan.get('reuse_source') else 'build/resume full history'))
    print(f'Cumulative batch calls: {used}/{a.max_total_calls}. Total sampled questions: {sum(len(p[1]["selected_question_indices"]) for p in plans)}')
    if not a.run:
        print('OFFLINE PLAN ONLY: no files modified, model imports, downloads or API calls.')
        return
    from deepseek_adapter import configure_session,RunStopped
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pilot.atomic_json(a.output_dir/'sampling_manifest.json',{'groups':[x[1] for x in plans],
        'dataset_sha256':hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pilot_sha256':hashlib.sha256((ROOT/'run_locomo_pilot.py').read_bytes()).hexdigest(),
        'adapter_sha256':hashlib.sha256((ROOT/'deepseek_adapter.py').read_bytes()).hexdigest()})
    combined=[]
    for args,plan,sample in plans:
        folder=Path(plan['run_directory']);folder.mkdir(parents=True,exist_ok=True)
        pilot.atomic_json(folder/'manifest.json',plan)
        if plan.get('reuse_source'):
            origin=Path(plan['reuse_source'])
            if not (folder/'memory_checkpoint.pkl').exists():shutil.copy2(origin/'memory_checkpoint.pkl',folder/'memory_checkpoint.pkl')
            if not (folder/'answers.jsonl').exists():
                existing=pilot.read_results(origin/'answers.jsonl',plan['fingerprint'])
                for i in plan['selected_question_indices']:pilot.append_jsonl(folder/'answers.jsonl',existing[i])
        existing=pilot.read_results(folder/'answers.jsonl',plan['fingerprint'])
        if all(i in existing for i in plan['selected_question_indices']):
            print('Reusing completed group',args.sample_index)
        else:
            ledger=folder/'usage.jsonl'
            own=sum(json.loads(s)['event']=='attempt' for s in ledger.read_text().splitlines() if s.strip()) if ledger.exists() else 0
            session=configure_session(ledger,max_calls=max(1,own+a.max_total_calls-used),max_output_tokens=1500)
            before=session.summary()['attempts']
            try:
                if used>=a.max_total_calls:raise RunStopped('Batch call cap reached; increase --max-total-calls to resume.')
                result=pilot.run_pilot(args,plan,sample,session,RunStopped)
                pilot.atomic_json(folder/'summary.json',{'status':'completed',**result,'usage':session.summary()})
            except (RunStopped,KeyboardInterrupt) as e:
                pilot.atomic_json(folder/'summary.json',{'status':'stopped','reason':str(e),'usage':session.summary()})
                print('STOPPED:',str(e));return
            finally:used+=session.summary()['attempts']-before
            existing=pilot.read_results(folder/'answers.jsonl',plan['fingerprint'])
        combined.extend(existing[i] for i in plan['selected_question_indices'])
    (a.output_dir/'answers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined))
    pilot.atomic_json(a.output_dir/'summary.json',{'status':'completed','questions':len(combined),
        'mean_metrics':{m:sum(r['metrics'][m] for r in combined)/len(combined) for m in ['f1','bleu1']},
        'new_batch_api_attempts':used,'note':'Equal-size uniform QA samples; full histories retained. Not a full benchmark evaluation.'})
    print('COMPLETED. Combined answers:',a.output_dir/'answers.jsonl')


if __name__=='__main__':main()
