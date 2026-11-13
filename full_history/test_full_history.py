"""Offline integration checks; no network calls or model downloads."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_full_history as runner
import deepseek_adapter as adapter


class ControlTests(unittest.TestCase):
    def test_all_turns_dates_captions_without_qa_leakage(self):
        sample = {'qa': [{'answer': 'SECRET_REFERENCE'}], 'session_summary': {'x': 'SECRET_SUMMARY'},
                  'conversation': {'session_1_date_time': '1 May 2023', 'session_1': [
                      {'speaker': 'A', 'text': 'hello', 'img_url': 'not downloaded', 'blip_caption': 'a tree'},
                      {'speaker': 'B', 'text': 'goodbye'}]}}
        history, turns = runner.full_history(sample)
        self.assertEqual(turns, 2)
        self.assertIn('1 May 2023', history)
        self.assertIn('A: [Image: a tree] hello', history)
        self.assertIn('B: goodbye', history)
        self.assertNotIn('SECRET', history)

    def test_pause_cap_resume_and_changed_configuration(self):
        plans = [dict(sample_index=0, sample_id='test', qa_index=i, question=f'q{i}', category=4,
                      reference='test', prompt='history question', raw_context='history',
                      temperature=0.7, max_output_tokens=1500) for i in range(3)]
        manifest = {'fingerprint': 'fixture', 'configuration': {'model': 'deepseek-flash'}}
        class FakeController:
            def __init__(self, model):
                self.model = model
            def get_completion(self, *args):
                call = adapter._session.begin(self.model)
                adapter._session.record(event='response', call_id=call, usage={})
                return '{"answer":"test"}'
        with tempfile.TemporaryDirectory() as temp, patch.object(adapter, 'DeepSeekController', FakeController), contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp)
            first = runner.execute(plans, manifest, output, 1)
            self.assertEqual((first['status'], first['completed_questions']), ('stopped', 1))
            second = runner.execute(plans, manifest, output, 5, max_new_questions=1)
            self.assertEqual((second['status'], second['completed_questions']), ('paused', 2))
            third = runner.execute(plans, manifest, output, 5)
            self.assertEqual((third['status'], third['completed_questions']), ('completed', 3))
            fourth = runner.execute(plans, manifest, output, 5)
            self.assertEqual(fourth['usage']['attempts'], 3)
            self.assertEqual(len(runner.read_rows(output/'answers.jsonl')), 3)
            changed = dict(manifest, fingerprint='changed')
            with self.assertRaises(ValueError):
                runner.execute(plans, changed, output, 5)


if __name__ == '__main__':
    unittest.main()
