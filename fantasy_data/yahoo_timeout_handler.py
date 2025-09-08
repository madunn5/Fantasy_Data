# fantasy_data/yahoo_timeout_handler.py
from typing import Optional
from yahoo_fantasy_api import yhandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import json
import logging

logger = logging.getLogger(__name__)

# (connect_timeout, read_timeout) in seconds
DEFAULT_TIMEOUT = (5, 20)

class TimeoutYHandler(yhandler.YHandler):
    """
    YHandler with:
      - per-request timeouts
      - retries (429/5xx) with backoff
      - robust JSON parsing
      - ensures Yahoo returns JSON via ?format=json
    """

    def __init__(self, sc, timeout: tuple = DEFAULT_TIMEOUT, total_retries: int = 3, backoff: float = 0.5):
        super().__init__(sc)
        self._timeout = timeout
        self._install_retries(total_retries, backoff)

    def _install_retries(self, total_retries: int, backoff: float):
        retry = Retry(
            total=total_retries,
            read=total_retries,
            connect=total_retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
        self.sc.session.mount("https://", adapter)
        self.sc.session.mount("http://", adapter)

    def _safe_headers(self, req_mkhdr: bool):
        if not req_mkhdr:
            return None
        try:
            return yhandler.YHandler._mk_headers(self)  # type: ignore[attr-defined]
        except Exception:
            return {"Accept": "application/json"}

    def get(self, uri: str, req_mkhdr: Optional[bool] = True):
        """
        Use session directly; enforce timeout, retries, and JSON format.
        """
        from yahoo_fantasy_api.yhandler import YAHOO_ENDPOINT

        url = f"{YAHOO_ENDPOINT}/{uri}"
        headers = self._safe_headers(bool(req_mkhdr))

        parse_attempts = 3
        delay = 0.4
        last_exc = None

        for attempt in range(1, parse_attempts + 1):
            # CRITICAL: ensure JSON response
            resp = self.sc.session.get(
                url,
                headers=headers,
                params={"format": "json"},
                timeout=self._timeout,
            )
            resp.raise_for_status()

            text = resp.text or ""
            if not text.strip():
                last_exc = ValueError("Empty response body")
            else:
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    snippet = text[:200].replace("\n", " ").strip()
                    last_exc = ValueError(f"Non-JSON response (status {resp.status_code}, len {len(text)}): {snippet}")
                except ValueError:
                    snippet = text[:200].replace("\n", " ").strip()
                    last_exc = ValueError(f"JSON parse error (status {resp.status_code}, len {len(text)}): {snippet}")

            logger.warning(f"[Yahoo GET parse retry {attempt}/{parse_attempts}] {url} -> {last_exc}")
            time.sleep(delay)

        raise RuntimeError(f"Yahoo API returned non-JSON for {url}: {last_exc}")