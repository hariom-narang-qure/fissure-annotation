from fissure_annotation.experiment import FLEX_TIMEOUT_SECONDS, portkey_client


def test_portkey_client_leaves_retries_to_retry_py(monkeypatch):
    "the SDK would retry a 429 once without a log line"
    monkeypatch.setenv("PORTKEY_API_KEY", "test-key")
    assert portkey_client().openai_client.max_retries == 0


def test_portkey_client_waits_as_long_as_flex(monkeypatch):
    monkeypatch.setenv("PORTKEY_API_KEY", "test-key")
    assert portkey_client().openai_client.timeout == FLEX_TIMEOUT_SECONDS

