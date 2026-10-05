"""A chat reply's `chart` block (CSV) becomes a line chart in the chat tab; the rest stays markdown (engine/charts.py)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import charts  # noqa: E402

REPLY = """Delivery speed fell in the last week of March.

```chart
Week,Avg hours,Orders
Mar 3,18,410
Mar 10,19,428
Mar 17,27,455
Mar 24,34,470
```

The drop starts on Mar 17, when the Leeds hub went to one shift."""


class ChartBlockTest(unittest.TestCase):
    def test_a_reply_is_split_into_text_and_a_chart(self):
        out = charts.parts(REPLY)
        self.assertEqual(["text", "chart", "text"], [k for k, _ in out])
        self.assertEqual("Delivery speed fell in the last week of March.", out[0][1].strip())
        frame = out[1][1]
        self.assertEqual(["Mar 3", "Mar 10", "Mar 17", "Mar 24"], list(frame.index))
        self.assertEqual(["Avg hours", "Orders"], list(frame.columns))
        self.assertEqual([18, 19, 27, 34], list(frame["Avg hours"]))
        self.assertTrue(out[2][1].strip().startswith("The drop starts on Mar 17"))

    def test_other_fences_and_plain_text_are_left_alone(self):
        self.assertEqual([("text", "Plain answer.")], charts.parts("Plain answer."))
        code = "```python\nprint(1)\n```"
        self.assertEqual([("text", code)], charts.parts(code))
        self.assertEqual([], charts.parts(""))

    def test_a_block_that_is_not_a_numeric_table_stays_visible_as_text(self):
        self.assertEqual([("code", "not,a,table")], charts.parts("```chart\nnot,a,table\n```"))
        one_row = "```chart\nDay,Value\nMon,3\n```"
        self.assertEqual([("code", "Day,Value\nMon,3")], charts.parts(one_row))
        words = "```chart\nDay,Mood\nMon,fine\nTue,great\n```"
        self.assertEqual("code", charts.parts(words)[0][0])

    def test_at_most_three_series_are_drawn(self):
        csv = "```chart\nx,a,b,c,d\n1,1,2,3,4\n2,2,3,4,5\n```"
        frame = charts.parts(csv)[0][1]
        self.assertEqual(["a", "b", "c"], list(frame.columns))


if __name__ == "__main__":
    unittest.main()
