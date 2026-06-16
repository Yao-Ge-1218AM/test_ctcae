"""Shared helpers for remote biological APIs and local TSV resources."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any


API_DIR = Path(__file__).resolve().parent
DEFAULT_TIMEOUT = float(os.getenv("GENEAGENT_REQUEST_TIMEOUT", "20"))
DEFAULT_RETRIES = int(os.getenv("GENEAGENT_REQUEST_RETRIES", "3"))
DEFAULT_BACKOFF = float(os.getenv("GENEAGENT_RETRY_BACKOFF", "1.0"))


def _requests_module() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The requests package is required for remote API calls."
        ) from exc
    return requests


def request(
    method: str,
    url: str,
    *,
    retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
    backoff: float = DEFAULT_BACKOFF,
    **kwargs: Any,
) -> tuple[Any | None, str | None]:
    """Request with timeout and exponential retry for transient failures."""

    requests = _requests_module()
    last_error: str | None = None

    for attempt in range(max(1, retries)):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            if response.status_code in {429} or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if attempt < retries - 1:
                    time.sleep(backoff * (2**attempt))
                    continue
            return response, None
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))

    return None, f"Error: {last_error or 'request failed'}"


def request_json(method: str, url: str, **kwargs: Any) -> tuple[Any | None, str | None]:
    response, error = request(method, url, **kwargs)
    if error:
        return None, error
    if response is None:
        return None, "Error: empty response"
    if not response.ok:
        return None, f"Error: HTTP {response.status_code}: {response.text[:200]}"
    try:
        return response.json(), None
    except ValueError as exc:
        return None, f"Error: invalid JSON response: {exc}"


def request_content(method: str, url: str, **kwargs: Any) -> tuple[bytes | None, str | None]:
    response, error = request(method, url, **kwargs)
    if error:
        return None, error
    if response is None:
        return None, "Error: empty response"
    if not response.ok:
        return None, f"Error: HTTP {response.status_code}: {response.text[:200]}"
    return response.content, None


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def resolve_db_path(dbpath: str | None, default_name: str) -> Path:
    candidate = Path(dbpath or default_name)
    if not candidate.is_absolute():
        candidate = API_DIR / candidate
    if not candidate.exists():
        local = API_DIR / candidate.name
        if local.exists():
            candidate = local
    if not candidate.exists():
        raise FileNotFoundError(f"Local database file not found: {candidate}")
    return candidate


def read_tsv_records(
    dbpath: str | None,
    default_name: str,
    required_columns: list[str],
) -> list[dict[str, str]]:
    path = resolve_db_path(dbpath, default_name)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]
