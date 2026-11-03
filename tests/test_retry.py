import pytest

from fissure_annotation.retry import (
    RETRY_TRANSIENT_ERRORS,
    WAIT_FOR_CREDIT,
    WAIT_FOR_FLEX_CAPACITY,
    RetrySettings,
    is_out_of_capacity,
    is_out_of_credit,
    is_transient,
    with_retries,
)


class StatusError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class APIConnectionError(Exception):
    "named like the OpenAI class, which is all is_transient looks at"


def failing(errors, result="ok"):
    "raises the given errors in turn, then returns the result"
    calls = []

    def request():
        calls.append(1)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return result

    return request, calls


@pytest.mark.parametrize("status", [408, 429, 500, 503, 504])
def test_timeouts_rate_limits_and_server_errors_are_transient(status):
    assert is_transient(StatusError(status))


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_client_errors_are_not_transient(status):
    assert not is_transient(StatusError(status))


def test_a_lost_connection_is_transient_and_a_plain_error_is_not():
    assert is_transient(APIConnectionError())
    assert not is_transient(ValueError())


def settings(**values) -> RetrySettings:
    return RetrySettings(**{"capacity_wait_seconds": 60.0, "credit_wait_seconds": 300.0, "retry_wait_seconds": 1.0, **values})


def test_a_transient_error_is_retried_until_the_request_succeeds():
    request, calls = failing([StatusError(503), StatusError(429)])
    waits = []
    assert with_retries(request, settings(retries=3), sleep=waits.append) == "ok"
    assert len(calls) == 3 and len(waits) == 2


def test_a_client_error_is_raised_at_once():
    request, calls = failing([StatusError(400)])
    with pytest.raises(StatusError):
        with_retries(request, settings(retries=3), sleep=lambda s: None)
    assert len(calls) == 1


def test_the_last_transient_error_is_raised_when_the_retries_run_out():
    request, calls = failing([StatusError(503)] * 5)
    with pytest.raises(StatusError):
        with_retries(request, settings(retries=2), sleep=lambda s: None)
    assert len(calls) == 3


def test_the_transient_wait_is_capped():
    request, _ = failing([StatusError(503)] * 20)
    waits = []
    with_retries(request, settings(retries=20, retry_wait_seconds=2.0), sleep=waits.append)
    assert max(waits) <= 60.0


class FlexUnavailable(Exception):
    "like the OpenAI RateLimitError for a lack of flex capacity"

    status_code = 429
    code = "flex_unavailable"


def test_a_lack_of_flex_capacity_is_recognised_by_its_code_or_its_message():
    assert is_out_of_capacity(FlexUnavailable())
    assert is_out_of_capacity(Exception("{'code': 'flex_unavailable'}"))
    assert not is_out_of_capacity(StatusError(429))
    assert not is_transient(FlexUnavailable())


def test_a_lack_of_capacity_is_waited_out_at_a_constant_rate_beyond_the_retries():
    request, calls = failing([FlexUnavailable()] * 10)
    waits = []
    assert with_retries(request, settings(retries=2), sleep=waits.append) == "ok"
    assert len(calls) == 11 and waits == [60.0] * 10


def test_waiting_for_capacity_does_not_use_up_the_retries_of_other_errors():
    request, _ = failing([FlexUnavailable(), FlexUnavailable(), StatusError(503), StatusError(503)])
    assert with_retries(request, settings(retries=2, retry_wait_seconds=0.0), sleep=lambda s: None) == "ok"


class OutOfCredit(StatusError):
    "what OpenAI answers when the organization reaches its spend limit"

    def __init__(self):
        super().__init__(429)
        self.code, self.type = "organization_spend_limit_exceeded", "insufficient_quota"


def test_an_empty_balance_or_a_spend_limit_is_recognised_and_is_not_transient():
    assert is_out_of_credit(OutOfCredit())
    assert is_out_of_credit(Exception("{'code': 'credit_balance_exhausted'}"))
    assert not is_transient(OutOfCredit())


def test_a_lack_of_credit_is_waited_out_with_doubling_waits_up_to_the_cap():
    request, calls = failing([OutOfCredit()] * 6)
    waits = []
    assert with_retries(request, settings(credit_wait_seconds=300.0, credit_max_wait_seconds=1200.0), sleep=waits.append) == "ok"
    assert waits == [300.0, 600.0, 1200.0, 1200.0, 1200.0, 1200.0]


def test_every_wait_is_reported_with_its_policy_attempt_and_wait():
    request, _ = failing([FlexUnavailable(), OutOfCredit(), StatusError(503)])
    reports = []
    with_retries(request, settings(retries=2), sleep=lambda s: None,
                 on_wait=lambda policy, error, attempt, wait: reports.append((policy, attempt, wait)))
    assert reports[0] == (WAIT_FOR_FLEX_CAPACITY, 1, 60.0)
    assert reports[1] == (WAIT_FOR_CREDIT, 1, 300.0)
    assert reports[2][:2] == (RETRY_TRANSIENT_ERRORS, 1)
