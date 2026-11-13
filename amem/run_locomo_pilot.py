"""Bounded, resumable A-MEM pilot. Default: inspect the plan without network access.

The upstream algorithm and user prompts are reused. The provider adapter uses
JSON mode with the original schema in the system prompt and local validation.
This is a small pilot, not a full paper reproduction. Configure .env first.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT


def configure_local_caches():
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    os.environ["NLTK_DATA"] = str(ROOT / ".cache" / "nltk_data")
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_results(path, fingerprint):
    results = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["fingerprint"] != fingerprint:
                raise ValueError("Results fingerprint mismatch; refusing to reuse results.")
            if row["qa_index"] in results:
                raise ValueError("Duplicate question in results; inspect the JSONL file.")
            results[row["qa_index"]] = row
    return results


def make_plan(args):
    dataset_path = Path(args.dataset).expanduser().resolve()
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not 0 <= args.sample_index < len(data):
        raise ValueError("sample-index must be between 0 and {}".format(len(data) - 1))
    sample = data[args.sample_index]
    sessions = [(key, value) for key, value in sample["conversation"].items()
                if key.startswith("session_") and isinstance(value, list)]
    for key, _ in sessions:
        if not sample["conversation"].get(key + "_date_time"):
            raise ValueError("Missing conversation timestamp: " + key)
    turns = sum(len(value) for _, value in sessions)
    selected = min(args.questions, len(sample["qa"]))
    source_names = ("memory_layer.py", "test_advanced.py", "load_dataset.py", "utils.py",
                    "deepseek_adapter.py", "run_pilot.py")
    source_hashes = {name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
                     if (SOURCE_ROOT / name).exists() else "missing" for name in source_names}
    configuration = {
        "model": args.model,
        "endpoint": "https://api.deepseek.com",
        "api_format": "json_object_with_schema_in_system_prompt_and_local_validation",
        "thinking": "disabled", "max_output_tokens": args.max_output_tokens,
        "sample_index": args.sample_index, "sample_id": sample.get("sample_id"),
        "sample_sha256": digest(sample), "source_sha256": source_hashes,
        "embedding": "all-MiniLM-L6-v2", "retrieve_k": args.retrieve_k,
        "temperature": 0.7, "temperature_c5": args.temperature_c5,
        "seed": args.seed, "refresh_index": False,
        "legacy_c5_reference": args.legacy_c5_reference,
        "metrics": ["upstream_set_token_f1", "upstream_bleu1"],
    }
    fingerprint = digest(configuration)
    run_dir = ROOT / "outputs" / "pilot" / fingerprint[:20]
    return {
        "pilot_only": True, "dataset": str(dataset_path), "fingerprint": fingerprint,
        "configuration": configuration, "complete_history_turns": turns,
        "sessions": len(sessions), "questions_requested": args.questions,
        "questions_selected": selected, "available_questions": len(sample["qa"]),
        "estimated_calls_without_cache": 2 * turns + 2 * selected,
        "max_calls_cumulative": args.max_calls, "run_directory": str(run_dir),
    }, sample


def print_plan(plan):
    print("A-MEM pilot only — not a formal/full reproduction.")
    print("Conversation {} ({}) | {} sessions | {} complete-history turns | {} / {} QA".format(
        plan["configuration"]["sample_index"], plan["configuration"]["sample_id"],
        plan["sessions"], plan["complete_history_turns"], plan["questions_selected"],
        plan["available_questions"]))
    print("Fresh-run estimate: 2 × {} turns + 2 × {} questions = {} API calls.".format(
        plan["complete_history_turns"], plan["questions_selected"],
        plan["estimated_calls_without_cache"]))
    print("Cumulative call cap: {}. Increase this explicitly to resume beyond the cap.".format(
        plan["max_calls_cumulative"]))
    print("Output / matching checkpoint: " + plan["run_directory"])
    print("No monetary estimate yet: usage records will report actual tokens and calls.")


def save_checkpoint(path, fingerprint, system, processed_turns, total_turns):
    # MemoryNote contains only note data. Never pickle the agent, controllers or API clients.
    state = {
        "fingerprint": fingerprint, "processed_turns": processed_turns,
        "complete": processed_turns == total_turns,
        "memories": {key: dict(note.__dict__) for key, note in system.memories.items()},
        "corpus": system.retriever.corpus, "embeddings": system.retriever.embeddings,
        "document_ids": system.retriever.document_ids, "evo_cnt": system.evo_cnt,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def restore_checkpoint(path, fingerprint, system, total_turns, note_class):
    if not path.exists():
        return 0
    # Only load this runner's locally generated checkpoints; pickle is not a download format.
    with path.open("rb") as handle:
        state = pickle.load(handle)
    if state["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint fingerprint mismatch.")
    done = state["processed_turns"]
    if not 0 <= done <= total_turns or len(state["memories"]) != done:
        raise ValueError("Checkpoint turn count mismatch.")
    if len(state["corpus"]) != done or (done and len(state["embeddings"]) != done):
        raise ValueError("Checkpoint retriever is incomplete.")
    if state["complete"] != (done == total_turns):
        raise ValueError("Checkpoint completion marker mismatch.")
    system.memories = {}
    for key, attributes in state["memories"].items():
        note = note_class.__new__(note_class)
        note.__dict__.update(attributes)
        system.memories[key] = note
    system.retriever.corpus = state["corpus"]
    system.retriever.embeddings = state["embeddings"]
    system.retriever.document_ids = state["document_ids"]
    system.evo_cnt = state["evo_cnt"]
    return done


def pilot_metrics(prediction, reference):
    # Exactly the F1 tokenization/set calculation in utils.calculate_metrics;
    # omit expensive BERTScore and other metrics in this pilot.
    from utils import simple_tokenize, calculate_bleu_scores
    if not prediction or not reference:
        return {"f1": 0.0, "bleu1": 0.0}
    prediction, reference = str(prediction).strip(), str(reference).strip()
    pred_tokens, ref_tokens = set(simple_tokenize(prediction)), set(simple_tokenize(reference))
    intersection = len(pred_tokens & ref_tokens)
    f1 = 2.0 * intersection / (len(pred_tokens) + len(ref_tokens)) if pred_tokens and ref_tokens else 0.0
    return {"f1": f1, "bleu1": calculate_bleu_scores(prediction, reference)["bleu1"]}


def run_pilot(args, plan, raw_sample, session, stopped_class):
    run_dir = Path(plan["run_directory"])
    checkpoint = run_dir / "memory_checkpoint.pkl"
    results_path = run_dir / "answers.jsonl"
    results = read_results(results_path, plan["fingerprint"])
    with (run_dir / "upstream.log").open("a", encoding="utf-8") as log:
        # Import only in --run mode: upstream imports can download small embedding/NLTK assets.
        with contextlib.redirect_stdout(log):
            import memory_layer
            from deepseek_adapter import DeepSeekController
            memory_layer.OpenAIController = DeepSeekController
            from test_advanced import advancedMemAgent
            from load_dataset import QA, parse_conversation
            import nltk
            try:
                nltk.word_tokenize("tokenizer readiness check")
            except LookupError:
                raise stopped_class("NLTK tokenizer resources missing; run prepare_resources.py first.") from None
            agent = advancedMemAgent(args.model, "openai", args.retrieve_k, args.temperature_c5,
                                     refresh_index=False)
        conversation = parse_conversation(raw_sample["conversation"])
        turns = [(section.date_time, turn) for section in conversation.sessions.values()
                 for turn in section.turns]
        if len(turns) != plan["complete_history_turns"]:
            raise ValueError("Parsed history differs from the preflight plan.")
        done = restore_checkpoint(checkpoint, plan["fingerprint"], agent.memory_system,
                                  len(turns), memory_layer.MemoryNote)
        memory_layer.FAILURES.clear()
        print("Resuming at turn {} / {}; {} QA results already stored.".format(done, len(turns), len(results)))
        for turn_index in range(done, len(turns)):
            date, turn = turns[turn_index]
            session.phase = "memory_build"
            with contextlib.redirect_stdout(log):
                agent.add_memory("Speaker " + turn.speaker + "says : " + turn.text, time=date)
            if any(memory_layer.FAILURES.values()):
                raise stopped_class("Upstream absorbed an error; current turn was NOT checkpointed: "
                                    + json.dumps(dict(memory_layer.FAILURES)))
            save_checkpoint(checkpoint, plan["fingerprint"], agent.memory_system,
                            turn_index + 1, len(turns))
            if (turn_index + 1) % 10 == 0 or turn_index + 1 == len(turns):
                print("Checkpointed history: {} / {} turns".format(turn_index + 1, len(turns)), flush=True)
        selected_indices = plan.get("selected_question_indices", list(range(plan["questions_selected"])))
        for qa_index in selected_indices:
            raw_qa = raw_sample["qa"][qa_index]
            if qa_index in results:
                continue
            qa = QA(question=raw_qa["question"], answer=raw_qa.get("answer"),
                    evidence=raw_qa.get("evidence", []), category=raw_qa.get("category"),
                    adversarial_answer=raw_qa.get("adversarial_answer"))
            session.phase = "qa"
            started = time.perf_counter()
            rng = random.Random("{}:{}:{}".format(args.seed, args.sample_index, qa_index))
            with contextlib.redirect_stdout(log):
                raw_prediction, prompt, context = agent.answer_question(qa.question, qa.category, qa.final_answer, rng)
            if any(memory_layer.FAILURES.values()):
                raise stopped_class("Upstream error during QA; answer was not saved.")
            prediction = json.loads(raw_prediction)["answer"]
            reference = qa.reference(args.legacy_c5_reference)
            metrics = pilot_metrics(prediction, reference)
            row = {
                "fingerprint": plan["fingerprint"], "sample_index": args.sample_index,
                "sample_id": raw_sample.get("sample_id"), "qa_index": qa_index,
                "question": qa.question, "category": qa.category,
                "prediction": prediction, "reference": reference, "metrics": metrics,
                "prompt": prompt, "raw_context": context,
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_jsonl(results_path, row)
            results[qa_index] = row
            print("Saved QA {} / {}".format(qa_index + 1, plan["questions_selected"]), flush=True)
    selected_ids = set(plan.get("selected_question_indices", range(plan["questions_selected"])))
    selected = [row for i, row in results.items() if i in selected_ids]
    return {"completed_questions": len(selected), "mean_metrics": {
        key: sum(row["metrics"][key] for row in selected) / len(selected)
        for key in ("f1", "bleu1")
    } if selected else {}}


def main():
    global SOURCE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--resume-dir", type=Path, help="Resume an existing experiment; verify model, dataset and core source hashes")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Run paid API requests with checkpoints")
    mode.add_argument("--check-api", action="store_true", help="Send exactly one paid JSON-format test request")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "locomo10.json"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--max-calls", type=int, default=100,
                        help="Cumulative attempt limit in this configuration's ledger")
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--temperature-c5", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-c5-reference", action="store_true",
                        help="Score C5 against the paper's legacy adversarial reference; default is corrected")
    args = parser.parse_args()
    SOURCE_ROOT = Path(args.source_dir).expanduser().resolve()
    sys.path.insert(0, str(SOURCE_ROOT))
    if min(args.questions, args.max_calls, args.max_output_tokens, args.retrieve_k) < 1:
        parser.error("questions, max-calls, max-output-tokens and retrieve-k must be positive")
    configure_local_caches()
    if args.run or args.check_api:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    plan, raw_sample = make_plan(args)
    if args.resume_dir:
        resume_dir = args.resume_dir.expanduser().resolve()
        original = json.loads((resume_dir / "manifest.json").read_text())
        old, new = original["configuration"], plan["configuration"]
        for key in set(old) | set(new):
            if key != "source_sha256" and old.get(key) != new.get(key):
                parser.error("Resume configuration mismatch: " + key)
        for name in ("memory_layer.py", "test_advanced.py", "load_dataset.py", "utils.py"):
            if old["source_sha256"].get(name) != new["source_sha256"].get(name):
                parser.error("Core source changed: " + name)
        if not (resume_dir / "memory_checkpoint.pkl").is_file():
            parser.error("Resume checkpoint missing")
        plan["configuration"] = old
        plan["fingerprint"] = original["fingerprint"]
        plan["run_directory"] = str(resume_dir)
        saved = read_results(resume_dir / "answers.jsonl", plan["fingerprint"])
        print("Saved answers: {}; remaining requested: {}".format(len(saved),
              sum(i not in saved for i in range(plan["questions_selected"]))))
    if args.check_api:
        check_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        plan["run_directory"] = str(Path(plan["run_directory"]) / "api_checks" / check_id)
        plan["max_calls_cumulative"] = 1
    print_plan(plan)
    if not args.run and not args.check_api:
        print("DRY RUN: no API calls, imports of model libraries, or downloads were performed.")
        return 0
    from deepseek_adapter import configure_session, DeepSeekController, RunStopped
    run_dir = Path(plan["run_directory"])
    run_dir.mkdir(parents=True, exist_ok=True)
    plan["transport_execution"] = {
        "adapter_sha256": hashlib.sha256((ROOT / "deepseek_adapter.py").read_bytes()).hexdigest(),
        "output_policy": "Initial configured limit; retry length-truncated responses by doubling up to 6000 tokens, within cumulative call cap. Original checkpoint identity preserved.",
    }
    atomic_json(run_dir / "manifest.json", plan)
    session = configure_session(run_dir / ("api_check_usage.jsonl" if args.check_api else "usage.jsonl"),
                                max_calls=1 if args.check_api else args.max_calls,
                                max_output_tokens=args.max_output_tokens)
    summary = {"pilot_only": True, "fingerprint": plan["fingerprint"], "status": "running"}
    exit_code = 0
    try:
        if args.check_api:
            session.phase = "api_check"
            response = DeepSeekController(args.model).get_completion(
                'Return a JSON object with exactly this value: {"ok": true}',
                response_format={"type": "json_schema", "json_schema": {
                    "name": "api_check", "strict": True, "schema": {
                        "type": "object", "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"], "additionalProperties": False}}}, temperature=0.7)
            if json.loads(response) != {"ok": True}:
                raise RunStopped("API test did not return the expected JSON object.")
            print("API JSON check passed (one request).")
        else:
            summary.update(run_pilot(args, plan, raw_sample, session, RunStopped))
        summary["status"] = "completed"
    except RunStopped as error:
        summary.update(status="stopped", reason=str(error))
        print("STOPPED: " + str(error))
        print("Only fully completed turns/questions were saved; reuse this configuration to resume.")
    except KeyboardInterrupt:
        summary.update(status="stopped", reason="Interrupted by user")
        print("Interrupted; the last completed turn/question remains saved.")
    except Exception as error:
        summary.update(status="failed", reason="{}: {}".format(type(error).__name__, error))
        print(summary["reason"], file=sys.stderr)
        exit_code = 1
    finally:
        summary["usage"] = session.summary()
        atomic_json(run_dir / ("api_check_summary.json" if args.check_api else "summary.json"), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
