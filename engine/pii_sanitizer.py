"""PII scan and standard-parameter injection for generated codebases.

The identity to scrub is the one running the app: its email address, its username (the address's local part)
and the project it runs in, all read from runtime settings (GCLOUD_ACCOUNT or the active gcloud account, and
the configured project). No address or domain is hard-coded or allow-listed: every email address is replaced.
Rules run in order and never rescan text an earlier rule already replaced. After redaction every file is
scanned again; anything left is reported as a finding.
"""
import json
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

from engine.config import Settings

AUDIT_FILE = "SECURITY_PII_AUDIT.json"
PROJECT_PLACEHOLDER = "${YOUR_GCP_PROJECT_ID}"
EMAIL_PLACEHOLDER = "${CONTACT_EMAIL}"
USER_PLACEHOLDER = "${DEVELOPER}"
IP_PLACEHOLDER = "${HOST_IP}"
PLACEHOLDER_RE = re.compile(r"(\$\{[A-Z0-9_]+\})")
MIN_USERNAME_LEN = 3  # shorter local parts would match inside ordinary words
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
NON_ROUTABLE_IPS = ("0.0.0.0", "127.0.0.1")  # bind addresses, not personal data
CHECKS = ["private keys", "OAuth tokens", "API keys", "local home paths", "email addresses",
          "GCP project IDs/numbers", "usernames", "IP addresses"]


class Rule(NamedTuple):
    label: str
    pattern: "re.Pattern[str]"
    replacement: str
    keep: Optional[Callable[[str], bool]] = None  # True = this match is not PII, leave it


def _not_an_address(ip: str) -> bool:
    return ip in NON_ROUTABLE_IPS or any(int(octet) > 255 for octet in ip.split("."))


SECRET_RULES = (
    Rule("Private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
         "${SERVICE_ACCOUNT_KEY}"),
    Rule("OAuth token", re.compile(r"ya29\.[0-9A-Za-z_-]+"), "${GOOGLE_OAUTH_ACCESS_TOKEN}"),
    Rule("API key", re.compile(r"AIza[0-9A-Za-z_-]{35}"), "${GOOGLE_API_KEY}"),
    Rule("Local home path", re.compile(r"(?:/Users|/home)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.&+-]+)*"), "./data"),
    Rule("Email address", EMAIL_RE, EMAIL_PLACEHOLDER),
)
IP_RULE = Rule("IP address", IPV4_RE, IP_PLACEHOLDER, keep=_not_an_address)


class PIISanitizer:
    def __init__(self, project_ids: Iterable[str] = (), user_identifiers: Iterable[str] = ()):
        """project_ids: project IDs / numbers to replace with PROJECT_PLACEHOLDER.
        user_identifiers: account emails; their local parts are scrubbed as usernames too."""
        self.project_ids = sorted({p for p in project_ids if p}, key=len, reverse=True)
        names = {ident.split("@")[0] for ident in user_identifiers if ident}
        self.usernames = sorted((n for n in names if len(n) >= MIN_USERNAME_LEN), key=len, reverse=True)
        self.rules: Tuple[Rule, ...] = (
            *SECRET_RULES,
            *(Rule("GCP project identifier", re.compile(rf"(?<![A-Za-z0-9-]){re.escape(p)}(?![A-Za-z0-9-])"),
                   PROJECT_PLACEHOLDER) for p in self.project_ids),
            *(Rule("Username", re.compile(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", re.I), USER_PLACEHOLDER)
              for n in self.usernames),
            IP_RULE,
        )

    @classmethod
    def for_settings(cls, settings: Settings) -> "PIISanitizer":
        """Sanitizer for the identity running the app: its account, project ID and project number."""
        return cls(project_ids=[settings.project_id, settings.project_number],
                   user_identifiers=[settings.gcloud_account])

    def sanitize_text(self, text: str) -> Tuple[str, List[str]]:
        """-> (clean text, what was redacted, e.g. "Username x2 -> ${DEVELOPER}")."""
        if not text:
            return "", []
        counts: Counter = Counter()
        s = str(text)
        for rule in self.rules:
            s = self._apply(rule, s, counts)
        return s, [f"{label} x{n} -> {replacement}" for (label, replacement), n in counts.items()]

    @staticmethod
    def _apply(rule: Rule, text: str, counts: Counter) -> str:
        """Apply one rule outside the placeholders already in the text (so, for example, a username never
        matches inside ${CONTACT_EMAIL})."""
        def replace(m: re.Match) -> str:
            if rule.keep and rule.keep(m.group(0)):
                return m.group(0)
            counts[(rule.label, rule.replacement)] += 1
            return rule.replacement

        parts = PLACEHOLDER_RE.split(text)  # odd indexes are placeholders
        return "".join(p if i % 2 else rule.pattern.sub(replace, p) for i, p in enumerate(parts))

    def sanitize_files(self, files: Dict[str, str], earlier: Iterable[str] = ()) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Redact every file, re-scan the result, and add the AUDIT_FILE report. `earlier` lists redactions a
        previous pass already applied to these files; the report keeps them. -> (files, audit)."""
        clean, redactions = {}, list(earlier)
        for name, content in files.items():
            if name == AUDIT_FILE:
                continue
            clean[name], found = self.sanitize_text(content)
            redactions += [f"{name}: {r}" for r in found]
        leftovers = [f"{name}: {r}" for name, c in clean.items() for r in self.sanitize_text(c)[1]]
        report = {
            "pii_audit_status": "PASSED" if not leftovers else "FAILED",
            "files_scanned_count": len(clean),
            "files_scanned": sorted(clean),
            "checks": CHECKS,
            "redactions_applied": list(dict.fromkeys(redactions)),
            "remaining_findings": leftovers,
        }
        clean[AUDIT_FILE] = json.dumps(report, indent=2)
        return clean, report
