import pytest


@pytest.fixture(autouse=True)
def _offline_weather(monkeypatch):
    # Tests never hit the network: the example agent's tool uses deterministic fixtures.
    monkeypatch.setenv("MFA_WEATHER_PROVIDER", "fixture")
