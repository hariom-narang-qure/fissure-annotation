"""What a model request does when the provider fails it. Three policies, each one tenacity policy.

| Policy | Matches | Does | Gives up |
|---|---|---|---|
| `wait_for_flex_capacity` | 429 `flex_unavailable` | waits `capacity_wait_seconds`, again and again | never |
| `wait_for_credit` | an empty balance or a spend limit, or a 402 | waits `credit_wait_seconds`, doubling up to `credit_max_wait_seconds` | never |
| `retry_transient_errors` | 408, any other 429, 5xx, a lost connection | waits `retry_wait_seconds`, doubling with jitter up to 60 s | after `retries` retries, then raises |

Why:
- A slice is one conversation, and an error in any slice stops the run of its whole scan.
  - One 503 from the provider (a connection timeout upstream of Portkey) failed a scan of 289 slices.
- Flex has no capacity for minutes at a time. That is not an error of the request, so it is waited out.
- A spend limit is lifted when the organization raises it, so it is waited out too, with longer and longer waits.
  - Runs stopped at the OpenAI organization's spend limit on 2026-10-30 and on 2026-11-03.

How:
- The policies wrap each other: transient errors outside, then credit, then flex capacity inside.
  - Each policy retries only the errors it matches and lets every other error through.
  - So waiting for capacity or credit never uses up a retry of a transient error.
- Errors are recognised by their status code, error code, type or class name.
  - So the OpenAI classes vendored by Portkey need no import.
- Any other error is raised at once. The scan list decides what to do with the scan (scan_list.py).
- Every wait is reported through `on_wait` with the policy's name, so a run shows why it is waiting.
  - That is also where notifications can hook in later.
- Retrying a request is safe: nothing is saved before a response arrives, and `previous_response_id` points at a response that exists.

Stopping a run:
- The waits sleep on one `STOPPING` event, not `time.sleep`, and no request starts once it is set.
- `stop_waits_on_ctrl_c` sets it on Ctrl-C, so a run stops at once.
  - Without it, a run waiting for credit ignored Ctrl-C for up to an hour.
  - The thread pool of the stacks waits for its threads when it exits, and every queued stack still started after that.
"""

import signal
import threading
from collections.abc import Callable

from pydantic import BaseModel, Field
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    stop_never,
    wait_exponential,
    wait_fixed,
    wait_random_exponential,
)

RETRIED_STATUS_CODES = frozenset({408, 429})
OUT_OF_CREDIT_STATUS = 402  # OpenRouter answers 402 Payment Required when its credits run out
RETRIED_ERROR_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})
NO_CAPACITY_CODE = "flex_unavailable"
OUT_OF_CREDIT_CODES = frozenset({"insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded"})
MAX_TRANSIENT_WAIT_SECONDS = 60.0

WAIT_FOR_FLEX_CAPACITY = "wait_for_flex_capacity"
WAIT_FOR_CREDIT = "wait_for_credit"
RETRY_TRANSIENT_ERRORS = "retry_transient_errors"

STOPPING = threading.Event()

OnWait = Callable[[str, BaseException, int, float], None]  # the policy, the error, the attempt, the wait in seconds


class RetrySettings(BaseModel):
    capacity_wait_seconds: float = Field(default=60.0, ge=0)
    credit_wait_seconds: float = Field(default=300.0, ge=0)  # the first wait for credit, which doubles each time
    credit_max_wait_seconds: float = Field(default=3600.0, ge=0)
    retries: int = Field(default=6, ge=0)  # retries of a transient error, after the first attempt
    retry_wait_seconds: float = Field(default=2.0, ge=0)  # the first wait for a transient error, which doubles each time


class RunStopped(KeyboardInterrupt):
    "a KeyboardInterrupt, so every caller treats it like Ctrl-C, and no policy or scan retry catches it"


def sleep_unless_stopping(seconds: float) -> None:
    if STOPPING.wait(seconds):
        raise RunStopped


def raise_if_stopping() -> None:
    if STOPPING.is_set():
        raise RunStopped


def stop_on_interrupt(signal_number: int, frame) -> None:
    STOPPING.set()
    raise KeyboardInterrupt


def stop_waits_on_ctrl_c() -> None:
    "a signal handler can only be set from the main thread, so a script calls this before it starts a run"
    signal.signal(signal.SIGINT, stop_on_interrupt)


def is_out_of_capacity(error: BaseException) -> bool:
    "the code is on the OpenAI error, and in its message when a proxy wraps it"
    return getattr(error, "code", None) == NO_CAPACITY_CODE or NO_CAPACITY_CODE in str(error)


def is_out_of_credit(error: BaseException) -> bool:
    "OpenAI puts insufficient_quota in the type and the reason in the code, so both are checked, and the message for a proxy"
    codes = {getattr(error, "code", None), getattr(error, "type", None)}
    if getattr(error, "status_code", None) == OUT_OF_CREDIT_STATUS:
        return True
    return bool(codes & OUT_OF_CREDIT_CODES) or any(code in str(error) for code in OUT_OF_CREDIT_CODES)


def is_transient(error: BaseException) -> bool:
    "a lack of capacity or credit is also a 429, but its own policy handles it"
    if is_out_of_capacity(error) or is_out_of_credit(error):
        return False
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in RETRIED_STATUS_CODES or status >= 500
    return type(error).__name__ in RETRIED_ERROR_NAMES


def describe(error: BaseException) -> str:
    parts = [type(error).__name__, getattr(error, "status_code", None), getattr(error, "code", None)]
    return " ".join(str(part) for part in parts if part)


def reporter(policy: str, on_wait: OnWait | None) -> Callable[[RetryCallState], None]:
    def report(state: RetryCallState) -> None:
        if on_wait is not None:
            on_wait(policy, state.outcome.exception(), state.attempt_number, state.next_action.sleep)

    return report


def wait_for_flex_capacity(settings: RetrySettings, on_wait: OnWait | None, sleep: Callable[[float], None]) -> Retrying:
    return Retrying(
        retry=retry_if_exception(is_out_of_capacity),
        wait=wait_fixed(settings.capacity_wait_seconds),
        stop=stop_never,
        before_sleep=reporter(WAIT_FOR_FLEX_CAPACITY, on_wait),
        sleep=sleep,
        reraise=True,
    )


def wait_for_credit(settings: RetrySettings, on_wait: OnWait | None, sleep: Callable[[float], None]) -> Retrying:
    return Retrying(
        retry=retry_if_exception(is_out_of_credit),
        wait=wait_exponential(multiplier=settings.credit_wait_seconds, max=settings.credit_max_wait_seconds),
        stop=stop_never,
        before_sleep=reporter(WAIT_FOR_CREDIT, on_wait),
        sleep=sleep,
        reraise=True,
    )


def retry_transient_errors(settings: RetrySettings, on_wait: OnWait | None, sleep: Callable[[float], None]) -> Retrying:
    "jitter keeps the parallel stacks of a run from retrying in lockstep"
    return Retrying(
        retry=retry_if_exception(is_transient),
        wait=wait_random_exponential(multiplier=settings.retry_wait_seconds, max=MAX_TRANSIENT_WAIT_SECONDS),
        stop=stop_after_attempt(settings.retries + 1),
        before_sleep=reporter(RETRY_TRANSIENT_ERRORS, on_wait),
        sleep=sleep,
        reraise=True,
    )


def with_retries[T](
    request: Callable[[], T],
    settings: RetrySettings,
    on_wait: OnWait | None = None,
    sleep: Callable[[float], None] = sleep_unless_stopping,
) -> T:
    raise_if_stopping()
    capacity = wait_for_flex_capacity(settings, on_wait, sleep)
    credit = wait_for_credit(settings, on_wait, sleep)
    transient = retry_transient_errors(settings, on_wait, sleep)
    return transient(lambda: credit(lambda: capacity(request)))
