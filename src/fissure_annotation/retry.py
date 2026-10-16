"""Retry a model request through transient failures of the provider.

Why:
- A slice is one conversation, and an error in any slice stops the run of its whole scan.
  - One 503 from the provider (a connection timeout upstream of Portkey) failed a scan of 289 slices.
- The flex tier of OpenAI answers 429 when it has no capacity, and the advice is to retry with backoff.

How:
- An error is transient when its HTTP status is 408, 429 or 5xx, or when the connection failed or timed out.
  - Errors are recognised by their status code or class name, so the OpenAI classes vendored by Portkey need no import.
- A transient error is retried up to `retries` times, waiting `base * 2 ** attempt` seconds, capped and with jitter.
  - Jitter keeps the parallel slices of a run from retrying in lockstep.
- Any other error, or the last transient one, is raised.
- Retrying a request is safe: nothing is saved before a response arrives, and `previous_response_id` points at a response that exists.
"""

import random
import time
from collections.abc import Callable

RETRIED_STATUS_CODES = frozenset({408, 429})
RETRIED_ERROR_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})
MAX_WAIT_SECONDS = 60.0


def is_transient(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in RETRIED_STATUS_CODES or status >= 500
    return type(error).__name__ in RETRIED_ERROR_NAMES


def wait_seconds(attempt: int, base_seconds: float) -> float:
    return min(base_seconds * 2**attempt, MAX_WAIT_SECONDS) * random.uniform(0.5, 1.0)


def with_retries[T](
    request: Callable[[], T],
    retries: int,
    base_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    for attempt in range(retries + 1):
        try:
            return request()
        except Exception as error:
            if attempt == retries or not is_transient(error):
                raise
            sleep(wait_seconds(attempt, base_seconds))
    raise AssertionError("unreachable, the last attempt returns or raises")
