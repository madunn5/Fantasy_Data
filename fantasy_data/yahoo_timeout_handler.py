# fantasy_data/yahoo_timeout_handler.py
from yahoo_fantasy_api import yhandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Default timeouts (connect, read) in seconds
DEFAULT_TIMEOUT = (5, 20)

class TimeoutYHandler(yhandler.YHandler):
    """
    YHandler with:
      - per-request timeouts
      - retries with exponential backoff for transient errors
    """

    def __init__(self, sc, timeout: tuple = DEFAULT_TIMEOUT, total_retries: int = 3, backoff: float = 0.5):
        super().__init__(sc)
        self._timeout = timeout
        self._install_retries(total_retries, backoff)

    def _install_retries(self, total_retries: int, backoff: float):
        # rauth exposes a requests.Session at self.sc.session
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

    def get(self, uri):
        """Override to inject timeout while preserving original behavior."""
        # Store original timeout if any
        original_timeout = getattr(self.sc.session, 'timeout', None)
        
        # Set our timeout
        self.sc.session.timeout = self._timeout
        
        try:
            # Call parent get method
            return super().get(uri)
        finally:
            # Restore original timeout
            if original_timeout is not None:
                self.sc.session.timeout = original_timeout
            else:
                if hasattr(self.sc.session, 'timeout'):
                    delattr(self.sc.session, 'timeout')