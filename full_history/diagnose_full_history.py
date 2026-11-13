"""Resume with response diagnostics only; request and validation behavior unchanged.

Use --max-new-questions 1 and a cumulative --max-calls equal to prior attempts + 1.
No API keys, headers, or client objects are logged.
"""
import deepseek_adapter as adapter
import run_full_history as runner


def instrument(controller):
    original_create = controller.client.chat.completions.create

    def create_with_diagnostics(*args, **kwargs):
        response = original_create(*args, **kwargs)
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice else None
        details = {'event': 'response_diagnostic',
                   'call_id': sum(e['event'] == 'attempt' for e in adapter._session.events),
                   'finish_reason': choice.finish_reason if choice else None,
                   'content': content,
                   'refusal': getattr(choice.message, 'refusal', None) if choice else None}
        import json
        from jsonschema import validate
        try:
            obj = json.loads(content or '')
            validate(obj, runner.SCHEMA['json_schema']['schema'])
            details['validation'] = 'passed'
        except (ValueError, TypeError) as exc:
            details['validation'] = type(exc).__name__ + ': ' + str(exc)
        except Exception as exc:
            # ValidationError messages contain only response/schema information.
            details['validation'] = type(exc).__name__ + ': ' + str(exc)
        adapter._session.record(**details)
        print('Response diagnostic: ' + json.dumps(details, ensure_ascii=False), flush=True)
        return response

    controller.client.chat.completions.create = create_with_diagnostics
    return controller


def main():
    original = adapter.DeepSeekController

    class DiagnosticController(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instrument(self)

    adapter.DeepSeekController = DiagnosticController
    runner.main()


if __name__ == '__main__':
    main()
