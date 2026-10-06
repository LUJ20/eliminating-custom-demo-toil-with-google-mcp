"""Offline tests for vertex.strip_fences / parse_json: the code fence of a model answer must end at the answer's
own closing fence, not at the first ``` inside the generated code (the Northwind code generator failure: a
prompt string holding a ```sql fence cut pipeline.py mid-string -> 'unterminated string literal')."""
import json
import unittest

from engine import brain, vertex

CODE_WITH_INNER_FENCE = '''"""Analytics agent.

Usage::

    ```bash
    python pipeline.py --dry-run
    ```
"""
import json
import os

CONFIG = json.load(open(os.path.join(os.path.dirname(__file__), "usecase_config.json")))
SQL_PROMPT = """Answer with one query in a fence:
```sql
SELECT 1
```
"""


def run(payload: dict) -> dict:
    return {"prompt": SQL_PROMPT, "input": payload["input"]}


if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv:
        print("1. Query: BigQuery", CONFIG["models"]["fast"]["model"])
    else:
        print(json.dumps(run({"input": sys.argv[1]})))
'''


class StripFencesTest(unittest.TestCase):
    def test_whole_answer_is_one_fence_with_fences_inside(self):
        self.assertEqual(vertex.strip_fences(f"```python\n{CODE_WITH_INNER_FENCE}```"), CODE_WITH_INNER_FENCE.strip())
        self.assertEqual(vertex.strip_fences(f"```python\n{CODE_WITH_INNER_FENCE}\n```\n"), CODE_WITH_INNER_FENCE.strip())

    def test_prose_then_a_fence_that_ends_the_answer(self):
        text = f"Here is the file:\n```python\n{CODE_WITH_INNER_FENCE}```"
        self.assertEqual(vertex.strip_fences(text), CODE_WITH_INNER_FENCE.strip())

    def test_fence_followed_by_prose_keeps_the_first_block(self):
        self.assertEqual(vertex.strip_fences("```json\n{\"a\": 1}\n```\nDone."), '{"a": 1}')
        text = "Sure:\n```python\nprint(1)\n```\nThis prints one. Run it with `python x.py`."
        self.assertEqual(vertex.strip_fences(text), "print(1)")

    def test_an_answer_that_ends_with_a_fence_is_one_block(self):
        text = "```python\nDOC = '''\n```bash\npython pipeline.py\n```\n'''\nprint(DOC)\n```"
        self.assertEqual(vertex.strip_fences(text), "DOC = '''\n```bash\npython pipeline.py\n```\n'''\nprint(DOC)")

    def test_unclosed_fence_and_plain_text(self):
        self.assertEqual(vertex.strip_fences("```python\nprint(1)\n"), "print(1)")
        self.assertEqual(vertex.strip_fences("print(1)"), "print(1)")
        self.assertEqual(vertex.strip_fences(""), "")
        self.assertEqual(vertex.strip_fences(None), "")

    def test_generated_code_with_an_inner_fence_validates(self):
        code = brain.validate_pipeline(f"```python\n{CODE_WITH_INNER_FENCE}```")
        self.assertIn("SELECT 1", code)
        self.assertTrue(code.endswith('print(json.dumps(run({"input": sys.argv[1]})))'))

    def test_parse_json_still_tolerates_fences_and_prose(self):
        self.assertEqual(vertex.parse_json("```json\n{\"a\": [1, 2]}\n```"), {"a": [1, 2]})
        self.assertEqual(vertex.parse_json("Result: {\"a\": 1}"), {"a": 1})
        self.assertEqual(json.dumps(vertex.parse_json("{\"s\": \"```\"}")), '{"s": "```"}')


if __name__ == "__main__":
    unittest.main()
