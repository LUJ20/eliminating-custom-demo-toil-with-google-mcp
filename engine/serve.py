"""Container entry point (Cloud Run): start picking the newest models in the background, then run the Streamlit
app in this same process.

A new instance starts with no model registry, and resolving it (Developer Knowledge MCP discovery, verification on
Vertex AI, golden set, feature pages) takes a few minutes. Starting at boot means the first visitor usually finds
the models ready; one who comes sooner waits for this run instead of starting a second one (ModelResolver.refresh).
Same process on purpose: the app shares this run's lock and registry.

    python -m engine.serve        # listens on $PORT (Cloud Run sets it), default 8080
"""
import logging
import os
import sys
import threading
from typing import List, Optional

from engine.config import ROOT, Settings
from engine.model_resolver import ModelResolver

logger = logging.getLogger("studio.serve")

STREAMLIT_FLAGS = ("--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none",
                   "--browser.gatherUsageStats=false")


def warm_up(settings: Optional[Settings] = None) -> None:
    """Resolve the models when the registry is empty or due. Never raises (it runs in a daemon thread): on failure
    the app resolves them on first use, as it would without the warm-up."""
    try:
        resolver = ModelResolver(settings)
        if resolver.is_empty() or resolver.is_stale():
            resolver.refresh(log=logger.info)
    except Exception:  # thread boundary: log with traceback instead of dying silently
        logger.exception("model warm-up failed; the app resolves the models on first use")


def streamlit_argv() -> List[str]:
    """`streamlit run app.py` on $PORT (Cloud Run sets it; default 8080)."""
    port = os.environ.get("PORT", "").strip()
    return ["streamlit", "run", os.path.join(ROOT, "app.py"), f"--server.port={port if port.isdigit() else '8080'}",
            *STREAMLIT_FLAGS]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    threading.Thread(target=warm_up, name="model-warmup", daemon=True).start()
    from streamlit.web import cli as streamlit_cli  # here, so importing this module (tests) stays light
    sys.argv = streamlit_argv()
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
