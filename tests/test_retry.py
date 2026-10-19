import pytest

from fissure_annotation.retry import is_out_of_capacity, is_transient, wait_seconds, with_retries


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


def test_a_transient_error_is_retried_until_the_request_succeeds():
    request, calls = failing([StatusError(503), StatusError(429)])
    waits = []
    assert with_retries(request, retries=3, base_seconds=1.0, capacity_wait_seconds=60.0, sleep=waits.append) == "ok"
    assert len(calls) == 3 and len(waits) == 2


def test_a_client_error_is_raised_at_once():
    request, calls = failing([StatusError(400)])
    with pytest.raises(StatusError):
        with_retries(request, retries=3, base_seconds=1.0, capacity_wait_seconds=60.0, sleep=lambda s: None)
    assert len(calls) == 1


def test_the_last_transient_error_is_raised_when_the_retries_run_out():
    request, calls = failing([StatusError(503)] * 5)
    with pytest.raises(StatusError):
        with_retries(request, retries=2, base_seconds=1.0, capacity_wait_seconds=60.0, sleep=lambda s: None)
    assert len(calls) == 3


def test_the_wait_doubles_and_is_capped():
    assert 1.0 <= wait_seconds(1, 2.0) <= 4.0
    assert wait_seconds(20, 2.0) <= 60.0


class FlexUnavailable(Exception):
    "like the OpenAI RateLimitError for a lack of flex capacity"

    status_code = 429
    code = "flex_unavailable"


def test_a_lack_of_flex_capacity_is_recognised_by_its_code_or_its_message():
    assert is_out_of_capacity(FlexUnavailable())
    assert is_out_of_capacity(Exception("{'code': 'flex_unavailable'}"))
    assert not is_out_of_capacity(StatusError(429))


def test_a_lack_of_capacity_is_retried_at_a_constant_rate_beyond_the_retries():
    request, calls = failing([FlexUnavailable()] * 10)
    waits = []
    assert with_retries(request, retries=2, base_seconds=1.0, capacity_wait_seconds=60.0, sleep=waits.append) == "ok"
    assert len(calls) == 11 and waits == [60.0] * 10


def test_waiting_for_capacity_does_not_use_up_the_retries_of_other_errors():
    request, calls = failing([FlexUnavailable(), FlexUnavailable(), StatusError(503), StatusError(503)])
    assert with_retries(request, retries=2, base_seconds=0.0, capacity_wait_seconds=0.0, sleep=lambda s: None) == "ok"


def test_each_retry_is_reported_with_its_wait():
    request, _ = failing([FlexUnavailable(), StatusError(503)])
    reports = []
    with_retries(request, retries=2, base_seconds=1.0, capacity_wait_seconds=60.0, sleep=lambda s: None,
                 on_retry=lambda error, which, wait: reports.append((which, wait)))
    assert reports[0] == ("no flex capacity, wait 1", 60.0)
    assert reports[1][0] == "retry 1/2"
