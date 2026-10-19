"""Retry a model request through transient failures of the provider.

Why:
- A slice is one conversation, and an error in any slice stops the run of its whole scan.
  - One 503 from the provider (a connection timeout upstream of Portkey) failed a scan of 289 slices.
- The flex tier of OpenAI answers 429 `flex_unavailable` when it has no capacity, sometimes for minutes.
  - Six retries over two minutes were not enough, and a scan failed after eight minutes of work.

How:
- A lack of flex capacity is retried at a constant interval, `capacity_wait_seconds`, for as long as it lasts.
  - It is not an error of the request, so giving up would only throw away the work of the scan.
- Any other transient error is retried up to `retries` times, waiting `base * 2 ** attempt` seconds, capped and with jitter.
  - An error is transient when its HTTP status is 408, 429 or 5xx, or when the connection failed or timed out.
  - Errors are recognised by their status code, error code or class name, so the OpenAI classes vendored by Portkey need no import.
  - Jitter keeps the parallel slices of a run from retrying in lockstep.
- Any other error, or the last transient one, is raised.
- Each retry is reported through `on_retry`, so a run shows why it is waiting.
- Retrying a request is safe: nothing is saved before a response arrives, and `previous_response_id` points at a response that exists.
"""

import random
import time
from collections.abc import Callable

RETRIED_STATUS_CODES = frozenset({408, 429})
RETRIED_ERROR_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})
NO_CAPACITY_CODE = "flex_unavailable"
MAX_WAIT_SECONDS = 60.0

OnRetry = Callable[[Exception, str, float], None]  # the error, which retry this is, and the wait in seconds


def is_transient(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in RETRIED_STATUS_CODES or status >= 500
    return type(error).__name__ in RETRIED_ERROR_NAMES


def is_out_of_capacity(error: Exception) -> bool:
    "the code is on the OpenAI error, and in its message when a proxy wraps it"
    return getattr(error, "code", None) == NO_CAPACITY_CODE or NO_CAPACITY_CODE in str(error)


def describe(error: Exception) -> str:
    parts = [type(error).__name__, getattr(error, "status_code", None), getattr(error, "code", None)]
    return " ".join(str(part) for part in parts if part)


def wait_seconds(attempt: int, base_seconds: float) -> float:
    return min(base_seconds * 2**attempt, MAX_WAIT_SECONDS) * random.uniform(0.5, 1.0)


def with_retries[T](
    request: Callable[[], T],
    retries: int,
    base_seconds: float,
    capacity_wait_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: OnRetry | None = None,
) -> T:
    attempt = 0  # counts the transient errors only, a lack of capacity never uses up a retry
    waited_for_capacity = 0
    while True:
        try:
            return request()
        except Exception as error:
            if is_out_of_capacity(error):
                waited_for_capacity += 1
                wait, which = capacity_wait_seconds, f"no flex capacity, wait {waited_for_capacity}"
            elif is_transient(error) and attempt < retries:
                attempt += 1
                wait, which = wait_seconds(attempt - 1, base_seconds), f"retry {attempt}/{retries}"
            else:
                raise
            if on_retry is not None:
                on_retry(error, which, wait)
            sleep(wait)
