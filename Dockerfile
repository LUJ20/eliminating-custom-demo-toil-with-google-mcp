# Gemini + MCP Use-Case Studio on Cloud Run (built by ./deploy.sh with `gcloud run deploy --source .`).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# ffmpeg joins the Veo shots of a video longer than one shot and reads the last frame each next shot starts from
# (engine/media.py). Without it the studio still runs, but films only the first shot of such a video.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Run as a non-root user that owns /app, where projects and the model registry are written.
RUN useradd --create-home --uid 10001 studio && chown -R studio:studio /app
USER studio

EXPOSE 8080
# engine/serve.py starts picking the newest models in the background, then runs Streamlit on $PORT
# (default 8080) in the same process, so the first visitor does not wait for a cold model registry.
CMD ["python", "-m", "engine.serve"]
