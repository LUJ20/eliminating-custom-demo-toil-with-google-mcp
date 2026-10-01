"""Google Developer Knowledge MCP client: the search_documents and get_documents tools over JSON-RPC/HTTPS.

Errors are raised, never returned as values: transport errors as requests exceptions, HTTP / JSON-RPC /
tool errors as McpError, so each caller decides whether a failure is fatal or best-effort.
"""
import itertools
import json
from typing import Any, Dict, List

import requests

from engine.common import ApiError
from engine.config import Settings, user_token

ENDPOINT = "https://developerknowledge.googleapis.com/mcp"
JSONRPC_INTERNAL_ERROR = -32603
_REQUEST_IDS = itertools.count(1)


class McpError(ApiError):
    """HTTP, JSON-RPC or tool error from the Developer Knowledge MCP server."""


class McpKnowledgeClient:
    def __init__(self, settings: Settings):
        self.s = settings

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """tools/call -> the tool's structured result."""
        token = user_token(self.s.gcloud_account)
        if not token:
            raise McpError(401, "No gcloud access token. Run: gcloud auth login")
        resp = requests.post(
            ENDPOINT, timeout=30,
            json={"jsonrpc": "2.0", "id": next(_REQUEST_IDS), "method": "tools/call",
                  "params": {"name": name, "arguments": arguments}},
            headers={"Authorization": f"Bearer {token}", "X-Goog-User-Project": self.s.project_id})
        if resp.status_code != 200:
            if resp.status_code == 401:
                user_token(self.s.gcloud_account, fresh=True)  # a retry picks up a fresh token
            raise McpError(resp.status_code, " ".join(resp.text.split())[:300])
        try:
            data = resp.json()
        except ValueError:
            raise McpError(502, "the MCP server returned a non-JSON response")
        if data.get("error"):
            err = data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
            status = 500 if err.get("code") == JSONRPC_INTERNAL_ERROR else 400
            raise McpError(status, f"{name}: {str(err.get('message', ''))[:300]}")
        result = data.get("result") or {}
        text = "".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))
        if result.get("isError"):
            raise McpError(404 if "not found" in text.lower() else 400, f"{name}: {text[:300]}")
        if isinstance(result.get("structuredContent"), dict):
            return result["structuredContent"]
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError:
            raise McpError(502, f"{name} returned non-JSON content")
        return parsed if isinstance(parsed, dict) else {}

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Chunks of official Google docs ({parent, content, ...}) that match `query`."""
        return self.call_tool("search_documents", {"query": query}).get("results", [])

    def get_documents(self, names: List[str]) -> List[Dict[str, Any]]:
        """Full pages ({name, title, content, ...}) for MCP document names."""
        return self.call_tool("get_documents", {"names": names}).get("documents", []) if names else []
