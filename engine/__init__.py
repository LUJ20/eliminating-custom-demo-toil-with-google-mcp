"""Gemini + MCP Use-Case Studio engine: model resolution, troubleshooting, build flow and publishing.

Modules
  config                Settings and credentials (environment > .env > gcloud config > defaults)
  common                API errors, UTC time, locked JSON / JSONL files, text helpers
  mcp_knowledge_client  Google Developer Knowledge MCP client
  vertex                Vertex AI REST calls and probes
  model_resolver        Model Resolver sub-agent (discover, verify, canary, watch, roll back, features)
  troubleshooter        Troubleshooter agent (classify, fix, retry, diagnose)
  brain                 Planner, code generator and judge prompts and validators
  usecase_synthesizer   Build flow: ground, plan, code, judge, package, deck
  pii_sanitizer         PII scan of generated packages
  deck_generator        Per-build 5-slide deck
  artifact_store        Publish to Google Drive or Cloud Storage
"""
