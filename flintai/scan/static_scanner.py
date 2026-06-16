"""
static_scanner.py — Layer 2b: Open-Source Static Analysis
Runs Bandit, detect-secrets, pip-audit, and custom OpenGrep rules
against the fetched repository files.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from ..schema import RepoFile
from .opengrep_resolver import find_opengrep_binary
from .schema import PackageInfo

logger = logging.getLogger(__name__)

# Use sys.executable so the current Python interpreter finds installed
# packages rather than looking for CLI tools on the system PATH.
_PY = sys.executable


@dataclass
class StaticFinding:
    tool: str  # bandit | opengrep | detect_secrets | pip_audit | internal
    rule_id: str
    severity: str
    message: str
    filepath: str
    line: int = 0
    evidence: str = ""
    cwe: str = ""


_OPENGREP_RULES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "agent_opengrep_rules.yaml"
)


def _load_opengrep_rules() -> str:
    with open(_OPENGREP_RULES_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ── Severity mapping ─────────────────────────────────────────────────────────

BANDIT_SEVERITY_MAP = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}

OPENGREP_SEVERITY_MAP = {"ERROR": "high", "WARNING": "medium", "INFO": "low"}


# ── Static analysis runners ──────────────────────────────────────────────────


def run_bandit(files_dir: str) -> list[StaticFinding]:
    """Run Bandit on Python files and parse findings."""
    findings = []
    try:
        result = subprocess.run(
            [
                _PY,
                "-m",
                "bandit",
                "-r",
                files_dir,
                "-f",
                "json",
                "-q",
                "--severity-level",
                "low",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if not result.stdout.strip():
            return findings

        data = json.loads(result.stdout)
        for issue in data.get("results", []):
            rel_path = issue.get("filename", "").replace(files_dir, "").lstrip("/\\")
            findings.append(
                StaticFinding(
                    tool="bandit",
                    rule_id=issue.get("test_id", ""),
                    severity=BANDIT_SEVERITY_MAP.get(
                        issue.get("issue_severity", "LOW"), "low"
                    ),
                    message=issue.get("issue_text", ""),
                    filepath=rel_path,
                    line=issue.get("line_number", 0),
                    evidence=issue.get("code", "")[:200],
                    cwe=issue.get("issue_cwe", {}).get("id", ""),
                )
            )
    except Exception as e:
        logger.error("Bandit error: %s", e)

    return findings


def run_opengrep(files_dir: str, rules_file: str) -> list[StaticFinding]:
    """Run OpenGrep with custom agent rules and parse findings."""
    findings = []
    opengrep_bin = find_opengrep_binary()
    if not opengrep_bin:
        logger.info(
            "OpenGrep not found — skipping pattern scan. "
            "Install from https://github.com/opengrep/opengrep/releases"
        )
        return findings
    try:
        result = subprocess.run(
            [
                opengrep_bin,
                "scan",
                "--config",
                rules_file,
                files_dir,
                "--json",
                "--quiet",
                "--no-git-ignore",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        # OpenGrep writes JSON to stdout on success (exit 0) and on
        # findings-found (exit 1). On rule errors (exit 7) it may
        # write to stderr instead. Try both.
        raw_json = result.stdout.strip() or result.stderr.strip()
        if not raw_json:
            return findings

        data = json.loads(raw_json)
        for err in data.get("errors", []):
            logger.warning("OpenGrep rule error: %s", err)
        for issue in data.get("results", []):
            rel_path = issue.get("path", "").replace(files_dir, "").lstrip("/\\")
            findings.append(
                StaticFinding(
                    tool="opengrep",
                    rule_id=issue.get("check_id", ""),
                    severity=OPENGREP_SEVERITY_MAP.get(
                        issue.get("extra", {}).get("severity", "INFO"), "low"
                    ),
                    message=issue.get("extra", {}).get("message", ""),
                    filepath=rel_path,
                    line=issue.get("start", {}).get("line", 0),
                    evidence=issue.get("extra", {}).get("lines", "")[:200],
                )
            )
    except Exception as e:
        logger.error("OpenGrep error: %s", e)

    return findings


def run_detect_secrets(files_dir: str) -> list[StaticFinding]:
    """Run detect-secrets to find hardcoded credentials."""
    findings = []
    try:
        result = subprocess.run(
            [_PY, "-m", "detect_secrets", "scan", files_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if not result.stdout.strip():
            return findings

        data = json.loads(result.stdout)
        for filepath, secrets in data.get("results", {}).items():
            rel_path = filepath.replace(files_dir, "").lstrip("/\\")
            for secret in secrets:
                findings.append(
                    StaticFinding(
                        tool="detect_secrets",
                        rule_id=secret.get("type", "secret"),
                        severity="high",
                        message=f"Potential secret detected: {secret.get('type', 'unknown')}",
                        filepath=rel_path,
                        line=secret.get("line_number", 0),
                        evidence=f"[redacted — line {secret.get('line_number', '?')}]",
                    )
                )
    except Exception as e:
        logger.error("detect-secrets error: %s", e)

    return findings


def _parse_pinned_packages(requirements_content: str) -> list[PackageInfo]:
    """
    Parse a requirements.txt and return a list of {name, version} dicts
    for all pinned (==) packages. Skips comments, blank lines, and unpinned entries.
    """
    packages = []
    for line in requirements_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" in line:
            # Strip inline comments
            line = line.split("#")[0].strip()
            parts = line.split("==")
            if len(parts) == 2:
                name = parts[0].strip()
                version = parts[1].strip().split()[0]  # strip any trailing extras
                if name and version:
                    packages.append({"name": name, "version": version})
    return packages


def _query_osv_api(packages: list[PackageInfo], filepath: str) -> list[StaticFinding]:
    """
    Query the OSV.dev batch API directly for a list of {name, version} packages.
    No virtual env, no package installation, no Python version constraints.
    Docs: https://osv.dev/docs/#tag/api/operation/OSV_QueryAffectedBatch
    """
    findings = []
    if not packages:
        return findings

    OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
    payload = {
        "queries": [
            {
                "version": pkg["version"],
                "package": {"name": pkg["name"], "ecosystem": "PyPI"},
            }
            for pkg in packages
        ]
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OSV_BATCH_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        logger.error("OSV API request failed: %s", e)
        return findings
    except Exception as e:
        logger.error("OSV API error: %s", e)
        return findings

    results = response.get("results", [])
    vuln_count = 0
    for i, result in enumerate(results):
        if i >= len(packages):
            break
        pkg = packages[i]
        vulns = result.get("vulns", [])
        for vuln in vulns:
            vuln_count += 1
            # Derive severity from CVSS score if available, else default to high
            severity = "high"
            cvss_score = None
            for severity_entry in vuln.get("severity", []):
                if severity_entry.get("type") == "CVSS_V3":
                    raw_score = severity_entry.get("score", 0)
                    # OSV may return a CVSS vector string instead of a numeric score
                    if isinstance(raw_score, str) and raw_score.startswith("CVSS:"):
                        break
                    try:
                        score = float(raw_score)
                        cvss_score = score  # noqa: F841
                        if score >= 9.0:
                            severity = "critical"
                        elif score >= 7.0:
                            severity = "high"
                        elif score >= 4.0:
                            severity = "medium"
                        else:
                            severity = "low"
                    except (ValueError, TypeError):
                        pass
                    break

            vuln_id = vuln.get("id", "")
            aliases = vuln.get("aliases", [])
            cve_id = next((a for a in aliases if a.startswith("CVE-")), vuln_id)
            summary = vuln.get("summary", "") or vuln.get("details", "")[:200]

            findings.append(
                StaticFinding(
                    tool="pip_audit",
                    rule_id=cve_id or vuln_id,
                    severity=severity,
                    message=(f"{pkg['name']}=={pkg['version']}: {summary[:200]}"),
                    filepath=filepath,
                    line=0,
                    evidence=(
                        f"Package: {pkg['name']} {pkg['version']} | "
                        f"ID: {vuln_id} | "
                        f"Aliases: {', '.join(aliases[:3]) if aliases else 'none'}"
                    ),
                )
            )

    logger.info(
        "OSV: %d CVE(s) found across %d of %d packages",
        vuln_count,
        len([p for p in results if p.get("vulns")]),
        len(packages),
    )
    return findings


def run_pip_audit(requirements_files: list) -> list[StaticFinding]:
    """
    Scan requirements files for known CVEs.
    Strategy:
      1. Try pip-audit (fast, uses local pip resolver)
      2. If pip-audit fails due to Python version conflicts or resolver errors,
         fall back to querying the OSV.dev batch API directly
         (no virtual env, no installation, no Python version constraints)
    """
    findings = []
    for req_file_path in requirements_files:
        if not req_file_path.endswith(".txt"):
            continue

        # ── Attempt 1: pip-audit ────────────────────────────────────────────────
        pip_audit_succeeded = False
        try:
            result = subprocess.run(
                [
                    _PY,
                    "-m",
                    "pip_audit",
                    "-r",
                    req_file_path,
                    "--format",
                    "json",
                    "-s",
                    "osv",
                    "--progress-spinner",
                    "off",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            # pip-audit exits 1 when vulnerabilities are found — that is normal.
            # Only treat return code > 1 as a hard error warranting fallback.
            if result.returncode <= 1:
                stdout = result.stdout.strip()
                if stdout:
                    data = json.loads(stdout)
                    dependencies = (
                        data if isinstance(data, list) else data.get("dependencies", [])
                    )
                    vuln_count = 0
                    for dep in dependencies:
                        for vuln in dep.get("vulns", []):
                            vuln_count += 1
                            findings.append(
                                StaticFinding(
                                    tool="pip_audit",
                                    rule_id=vuln.get("id", ""),
                                    severity="high",
                                    message=(
                                        f"{dep.get('name')}=={dep.get('version')}: "
                                        f"{vuln.get('description', '')[:200]}"
                                    ),
                                    filepath=req_file_path,
                                    line=0,
                                    evidence=(
                                        f"Package: {dep.get('name')} {dep.get('version')} | "
                                        f"CVE/ID: {vuln.get('id', 'N/A')} | "
                                        f"Fix: {vuln.get('fix_versions', [])}"
                                    ),
                                )
                            )
                    logger.info(
                        "pip-audit: %d CVE(s) found in %s",
                        vuln_count,
                        req_file_path.split("/")[-1],
                    )
                    pip_audit_succeeded = True
                else:
                    # Empty stdout — fall through to OSV fallback
                    stderr_preview = result.stderr.strip()[:200]
                    logger.warning(
                        "pip-audit: empty output — falling back to OSV API. Reason: %s",
                        stderr_preview,
                    )
            else:
                stderr_preview = result.stderr.strip()[:200]
                logger.warning(
                    "pip-audit exited %d — falling back to OSV API. Reason: %s",
                    result.returncode,
                    stderr_preview,
                )

        except json.JSONDecodeError as e:
            logger.warning(
                "pip-audit JSON parse error — falling back to OSV API: %s", e
            )
        except Exception as e:
            logger.warning("pip-audit error — falling back to OSV API: %s", e)

        # ── Attempt 2: OSV.dev direct API fallback ───────────────────────────────
        if not pip_audit_succeeded:
            try:
                content = open(req_file_path, "r", encoding="utf-8").read()
                packages = _parse_pinned_packages(content)
                if packages:
                    logger.info(
                        "OSV fallback: querying %d pinned packages from %s...",
                        len(packages),
                        req_file_path.split("/")[-1],
                    )
                    findings.extend(_query_osv_api(packages, req_file_path))
                else:
                    logger.info(
                        "OSV fallback: no pinned packages found in %s",
                        req_file_path.split("/")[-1],
                    )
            except Exception as e:
                logger.error("OSV fallback error: %s", e)

    return findings


def check_unpinned_dependencies(
    requirements_content: str, filepath: str
) -> list[StaticFinding]:
    """Check for unpinned AI framework dependencies."""
    findings = []
    ai_packages = [
        "crewai",
        "autogen",
        "pyautogen",
        "langchain",
        "openai",
        "anthropic",
        "langgraph",
        "ag2",
        "autogen-agentchat",
    ]

    for line in requirements_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg_name = re.split(r"[>=<!~\s]", line)[0].lower()
        if pkg_name in ai_packages:
            # Check if pinned (==) or unpinned (>=, ~=, no version)
            if "==" not in line:
                findings.append(
                    StaticFinding(
                        tool="internal",
                        rule_id="unpinned-ai-dependency",
                        severity="medium",
                        message=f"Unpinned AI framework dependency: '{line}' — supply chain risk",  # noqa: B950
                        filepath=filepath,
                        line=0,
                        evidence=line,
                    )
                )

    return findings


# ── Main static scanner entry point ─────────────────────────────────────────


def run_static_scan(
    python_files: list[RepoFile], requirements_files: list[RepoFile], tmp_dir: str
) -> list[StaticFinding]:
    """
    Write files to temp dir and run all static analysis tools.
    Returns combined list of StaticFinding objects.
    """
    all_findings = []

    # Write Python files to disk preserving their original relative
    # paths. Tools then report findings with paths relative to py_dir,
    # which match the original repo_file.path — no remapping needed.
    py_dir = os.path.join(tmp_dir, "src")
    os.makedirs(py_dir, exist_ok=True)

    for repo_file in python_files:
        dest = os.path.join(py_dir, repo_file.path.lstrip(os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(repo_file.content)

    # Write OpenGrep rules
    rules_path = os.path.join(tmp_dir, "agent_rules.yaml")
    with open(rules_path, "w") as f:
        f.write(_load_opengrep_rules())

    # Write requirements files preserving original paths.
    req_dir = os.path.join(tmp_dir, "reqs")
    os.makedirs(req_dir, exist_ok=True)
    req_paths = []
    for repo_file in requirements_files:
        if repo_file.path.endswith(".txt"):
            dest = os.path.join(req_dir, repo_file.path.lstrip(os.sep))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(repo_file.content)
            req_paths.append(dest)
            # Also check for unpinned deps inline
            all_findings.extend(
                check_unpinned_dependencies(repo_file.content, repo_file.path)
            )

    logger.info("Running Bandit on %d files...", len(python_files))
    all_findings.extend(run_bandit(py_dir))

    logger.info("Running OpenGrep with custom agent rules...")
    all_findings.extend(run_opengrep(py_dir, rules_path))

    logger.info("Running detect-secrets...")
    all_findings.extend(run_detect_secrets(py_dir))

    if req_paths:
        logger.info("Running pip-audit on %d requirements files...", len(req_paths))
        pip_findings = run_pip_audit(req_paths)
        for f in pip_findings:
            f.filepath = f.filepath.replace(req_dir, "").lstrip("/\\")
        all_findings.extend(pip_findings)

    logger.info("Total static findings: %d", len(all_findings))
    return all_findings
