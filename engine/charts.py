"""Charts in chat replies. The media director tells a chat assistant to return a requested trend as a fenced code
block tagged `chart` holding CSV (first column the x-axis labels, then one to three numeric columns). parts()
splits a reply into markdown text and such charts, so the UI (app.py) can draw them with st.line_chart and keep the
rest as markdown; a block that is not a readable numeric table stays visible as plain text."""
import io
import re
from typing import List, Tuple

CHART_BLOCK = re.compile(r"```chart[^\n]*\n(.*?)```", re.DOTALL)
MAX_SERIES = 3
MIN_POINTS = 2


def parts(text: str) -> List[Tuple[str, object]]:
    """-> [("text", markdown) | ("chart", DataFrame indexed by the first column) | ("code", raw block)], in order.
    Empty text parts are dropped."""
    out: List[Tuple[str, object]] = []
    text = text or ""
    pos = 0
    for m in CHART_BLOCK.finditer(text):
        if text[pos:m.start()].strip():
            out.append(("text", text[pos:m.start()]))
        pos = m.end()
        raw = m.group(1).strip()
        frame = table(raw)
        out.append(("chart", frame) if frame is not None else ("code", raw))
    if text[pos:].strip():
        out.append(("text", text[pos:]))
    return out


def table(csv_text: str):
    """The CSV as a DataFrame of numeric series indexed by its first column, or None when it is not one (fewer
    than two rows, no numeric column, or unreadable). pandas is imported here: the engine stays light without it."""
    try:
        import pandas as pd
        frame = pd.read_csv(io.StringIO(csv_text))
        if frame.shape[1] < 2 or len(frame) < MIN_POINTS:
            return None
        frame = frame.set_index(frame.columns[0])
        numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        if numeric.empty or numeric.isna().all(axis=None):
            return None
        return numeric.iloc[:, :MAX_SERIES]
    except Exception:  # any malformed block: shown as text instead
        return None
