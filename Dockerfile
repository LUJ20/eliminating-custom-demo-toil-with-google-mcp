# Gemini + MCP Use-Case Studio on Cloud Run (built by ./deploy.sh with `gcloud run deploy --source .`).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Run as a non-root user that owns /app, where projects and the model registry are written.
RUN useradd --create-home --uid 10001 studio && chown -R studio:studio /app
USER studio

EXPOSE 8080
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]
