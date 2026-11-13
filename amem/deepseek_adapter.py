"""DeepSeek transport adapter. Original A-MEM algorithms are not modified.

RunStopped deliberately inherits BaseException so upstream broad Exception
handlers cannot convert a budget/API failure into apparently valid memories.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


class RunStopped(BaseException):
    """Stop the experiment and keep its last completed checkpoint."""


class Session:
    def __init__(self, ledger_path: Path, max_calls: int, max_output_tokens: int):
        if max_calls < 1 or max_output_tokens < 1:
            raise ValueError("Call and output limits must be positive")
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.max_output_tokens = max_output_tokens
        self.phase = "unspecified"
        self.events = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.events.append(json.loads(line))

    def record(self, **event):
        event.update(timestamp=datetime.now(timezone.utc).isoformat(), phase=self.phase)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.events.append(event)

    def begin(self, model):
        attempts = sum(e["event"] == "attempt" for e in self.events)
        if attempts >= self.max_calls:
            raise RunStopped(f"已达到此实验累计调用上限 {self.max_calls}；未发送新请求。")
        call_id = attempts + 1
        # Persist BEFORE sending so a process interruption cannot erase an attempt.
        self.record(event="attempt", call_id=call_id, requested_model=model)
        return call_id

    def summary(self):
        responses = [e for e in self.events if e["event"] == "response"]
        attempts = [e for e in self.events if e["event"] == "attempt"]
        finished = {e.get("call_id") for e in self.events if e["event"] in ("response", "error")}
        return {
            "attempts": len(attempts),
            "successful_calls": len(responses),
            "prompt_tokens": sum(e.get("usage", {}).get("prompt_tokens", 0) or 0 for e in responses),
            "completion_tokens": sum(e.get("usage", {}).get("completion_tokens", 0) or 0 for e in responses),
            "token_usage_missing": sum(not e.get("usage") for e in responses),
            "unknown_outcome_attempts": sum(e["call_id"] not in finished for e in attempts),
            "errors": sum(e["event"] in ("error", "invalid_output") for e in self.events),
            "attempts_by_phase": dict(Counter(e["phase"] for e in attempts)),
            "max_calls": self.max_calls,
            "max_output_tokens": self.max_output_tokens,
            "billing_note": "Token usage from returned responses only; failed/interrupted requests may still incur charges. Check provider billing.",
        }


_session = None


def configure_session(ledger_path: Path, max_calls: int = 100, max_output_tokens: int = 1000):
    global _session
    _session = Session(ledger_path, max_calls, max_output_tokens)
    return _session


class DeepSeekController:
    def __init__(self, model: str, api_key=None):
        from openai import OpenAI
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key or key.strip() in {"your-key-here", "replace-me"}:
            raise RunStopped("未配置 DeepSeek key。先运行 configure_key.py；不要把 key 发到聊天里。")
        if _session is None:
            raise RuntimeError("Call configure_session before constructing a controller")
        self.model = model
        self.client = OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
            max_retries=0,
            timeout=90.0,
        )

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7):
        from jsonschema import validate, ValidationError
        schema = response_format.get("json_schema", {}).get("schema")
        system = "You must respond with a JSON object."
        if schema:
            # DeepSeek JSON mode does not enforce an OpenAI strict JSON schema.
            # Include the existing schema in the prompt and validate it locally.
            system += " The JSON object must satisfy this schema: " + json.dumps(schema)
        for network_attempt in range(3):
            call_id = _session.begin(self.model)
            start = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=_session.max_output_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                _session.record(event="error", call_id=call_id, error_type=type(exc).__name__,
                                status_code=status, seconds=time.monotonic() - start)
                transient = (status is not None and 500 <= status <= 599) or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
                if transient and network_attempt < 2:
                    delay = 5 * (network_attempt + 1)
                    _session.record(event="transient_retry", call_id=call_id, delay_seconds=delay)
                    print(f"Temporary API error; retrying in {delay}s ({network_attempt + 1}/2).", flush=True)
                    time.sleep(delay)
                    continue
                raise RunStopped(f"API request failed ({type(exc).__name__}, status={status}); stopped after bounded retries where applicable.") from None
        usage = response.usage.model_dump() if response.usage else {}
        choice = response.choices[0] if response.choices else None
        _session.record(event="response", call_id=call_id, returned_model=response.model,
                        usage=usage, seconds=time.monotonic() - start,
                        finish_reason=choice.finish_reason if choice else None)
        # Retry only a length-truncated response, with the same prompt and schema.
        # All attempts remain subject to the cumulative call limit and usage ledger.
        if choice is not None and choice.finish_reason == "length":
            previous_limit = _session.max_output_tokens
            if previous_limit < 6000:
                expanded_limit = min(6000, previous_limit * 2)
                _session.record(event="truncation_retry", call_id=call_id,
                                previous_limit=previous_limit, expanded_limit=expanded_limit)
                _session.max_output_tokens = expanded_limit
                try:
                    return self.get_completion(prompt, response_format, temperature)
                finally:
                    _session.max_output_tokens = previous_limit
        try:
            if choice is None or choice.finish_reason != "stop":
                raise ValueError("Incomplete response")
            content = choice.message.content
            obj = json.loads(content or "")
            if not isinstance(obj, dict):
                raise ValueError("Not a JSON object")
            if schema:
                validate(obj, schema)
        except (ValueError, TypeError, ValidationError):
            _session.record(event="invalid_output", call_id=call_id)
            raise RunStopped("模型输出为空、被截断或字段不符合要求；已停止，避免把失败结果写入记忆。") from None
        return content
