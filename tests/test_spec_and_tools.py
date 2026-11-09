import json
from datetime import date
from unittest import mock

import pytest

from mfa.registry import load_agent
from mfa.spec import AGENT_ENGINE, CLAUDE_MANAGED, CLOUD_RUN, LOCAL

from agents.weather_agent import weather


def test_agent_loads_by_short_name_and_module_ref():
    a = load_agent("weather_agent")
    b = load_agent("agents.weather_agent.agent:AGENT")
    assert a is b
    assert [t.name for t in a.tools] == ["get_weather_today"]
    for target in (LOCAL, AGENT_ENGINE, CLOUD_RUN, CLAUDE_MANAGED):
        assert a.model_for(target)


def test_tool_schema_is_derived_from_signature_and_docstring():
    t = load_agent("weather_agent").tool_by_name("get_weather_today")
    assert t.input_schema["required"] == ["location"]
    assert "Portland, Maine" in t.input_schema["properties"]["location"]["description"]
    assert "title" not in t.input_schema


def test_fixture_provider_is_deterministic():
    t = load_agent("weather_agent").tool_by_name("get_weather_today")
    out = json.loads(t.invoke({"location": "Paris, France"}))
    assert out["found"] and out["location"]["country"] == "France"
    assert out["date"] == date.today().isoformat()
    assert out["today"]["high_c"] == 15.0 and out["today"]["high_f"] == 59.0
    assert json.loads(t.invoke({"location": "Nowhereville"}))["found"] is False
    assert "other_matches" in json.loads(t.invoke({"location": "portland"}))


def _fake_open_meteo(url, params):
    if "geocoding" in url:
        if params["name"] == "Nowhere":
            return {}
        return {"results": [
            {"name": "Portland", "admin1": "Oregon", "country": "United States", "country_code": "US",
             "latitude": 45.5, "longitude": -122.7, "timezone": "America/Los_Angeles"},
            {"name": "Portland", "admin1": "Maine", "country": "United States", "country_code": "US",
             "latitude": 43.7, "longitude": -70.3, "timezone": "America/New_York"},
        ]}
    assert params["timezone"] == "auto" and params["forecast_days"] == 1
    return {
        "current": {"temperature_2m": 10.0, "apparent_temperature": 8.0, "weather_code": 3, "wind_speed_10m": 12.0},
        "daily": {"time": ["2026-11-07"], "weather_code": [61], "temperature_2m_max": [12.0],
                  "temperature_2m_min": [4.0], "precipitation_probability_max": [80]},
    }


def test_open_meteo_parsing_and_disambiguation():
    with mock.patch.object(weather, "_get_json", side_effect=_fake_open_meteo):
        r = weather.open_meteo("Portland, Maine")
        assert r["location"]["admin1"] == "Maine"
        assert r["other_matches"] == ["Portland, Oregon, United States"]
        assert r["today"] == {"conditions": "slight rain", "high_c": 12.0, "high_f": 53.6, "low_c": 4.0,
                              "low_f": 39.2, "precipitation_chance_pct": 80}
        assert r["current"]["conditions"] == "overcast" and r["date"] == "2026-11-07"
        assert weather.open_meteo("Nowhere")["found"] is False


def test_open_meteo_api_key_switches_to_customer_hosts(monkeypatch):
    seen: list[tuple[str, dict]] = []

    def recording(url, params):
        seen.append((url, params))
        return _fake_open_meteo(url, params)

    # Free hosts, no key: nothing is added to the query.
    with mock.patch.object(weather, "_get_json", side_effect=recording):
        assert weather.open_meteo("Portland, Oregon")["found"] is True
    assert [u for u, _ in seen] == [weather.GEOCODE_URL, weather.FORECAST_URL]
    assert all("apikey" not in p for _, p in seen)

    # A key moves both calls to the customer hosts and rides along on each request.
    seen.clear()
    monkeypatch.setenv("MFA_OPEN_METEO_API_KEY", "k-123")
    with mock.patch.object(weather, "_get_json", side_effect=recording):
        assert weather.open_meteo("Portland, Oregon")["found"] is True
    assert [u for u, _ in seen] == [weather.CUSTOMER_GEOCODE_URL, weather.CUSTOMER_FORECAST_URL]
    assert all(p["apikey"] == "k-123" for _, p in seen)

    # Explicit URLs win (a self-hosted instance), with or without a key.
    seen.clear()
    monkeypatch.delenv("MFA_OPEN_METEO_API_KEY")
    monkeypatch.setenv("MFA_OPEN_METEO_FORECAST_URL", "http://localhost:8080/v1/forecast")
    with mock.patch.object(weather, "_get_json", side_effect=recording):
        weather.open_meteo("Portland")
    assert [u for u, _ in seen] == [weather.GEOCODE_URL, "http://localhost:8080/v1/forecast"]
    assert all("apikey" not in p for _, p in seen)


def test_live_provider_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("MFA_WEATHER_PROVIDER", "open-meteo")
    with mock.patch.object(weather, "_get_json", side_effect=OSError("network down")):
        out = weather.today("Paris")
    assert out["found"] is False and "unavailable" in out["error"]


def test_mfa_specific_key_takes_precedence(monkeypatch):
    from mfa.credentials import anthropic_client
    from mfa.runtime.models import make_chat_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-generic")
    monkeypatch.setenv("MFA_ANTHROPIC_API_KEY", "sk-mfa")
    assert anthropic_client().api_key == "sk-mfa"
    assert make_chat_model("anthropic:claude-opus-5-5").anthropic_api_key.get_secret_value() == "sk-mfa"
    monkeypatch.delenv("MFA_ANTHROPIC_API_KEY")
    assert anthropic_client().api_key == "sk-generic"


def test_skill_folders_load_and_validate(tmp_path):
    from mfa.spec import AgentSpec, anthropic_skill, local_skills, skill, tool

    sk = skill("agents/weather_agent/skills/weather-briefing")
    assert sk.name == "weather-briefing" and sk.source == "custom" and sk.license == "Apache-2.0"
    assert sk.metadata == {"author": "multi-framework-agent", "version": "1.0"} and sk.has_scripts
    assert [rel for rel, _ in sk.files()] == ["SKILL.md", "references/clothing-guide.md", "scripts/briefing_lines.py"]
    assert len(sk.content_hash()) == 64
    assert [s.name for s in local_skills("agents/weather_agent/skills")] == ["weather-briefing"]
    assert local_skills(tmp_path / "nope") == []

    bad = tmp_path / "my-skill"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: other-name\ndescription: x\n---\nbody")
    with pytest.raises(ValueError, match="must match its folder name"):
        skill(bad)
    (bad / "SKILL.md").write_text("---\nname: My_Skill\ndescription: x\n---\nbody")
    with pytest.raises(ValueError, match="lowercase"):
        skill(bad)
    (bad / "SKILL.md").write_text("---\nname: my-skill\ndescription: uses <tags>\n---\nbody")
    with pytest.raises(ValueError, match="'<' or '>'"):
        skill(bad)
    (bad / "SKILL.md").write_text("---\nname: my-skill\ndescription: fine\nextra: nope\n---\nbody")
    with pytest.raises(ValueError, match="unexpected keys"):
        skill(bad)
    with pytest.raises(FileNotFoundError):
        skill(tmp_path)

    a = anthropic_skill("xlsx", version="2")
    assert (a.source, a.name, a.version, a.is_local, a.files()) == ("anthropic", "xlsx", "2", False, [])

    @tool
    def read(file_path: str) -> str:
        """Clashes with the built-in toolset.

        Args:
            file_path: A path.
        """
        return file_path

    with pytest.raises(ValueError, match="reserved for the built-in toolset"):
        AgentSpec(name="x", description="d", system_prompt="p", tools=[read])
    with pytest.raises(ValueError, match="unique"):
        AgentSpec(name="x", description="d", system_prompt="p", tools=[], skills=[sk, sk])


def test_open_meteo_retries_rate_limits(monkeypatch):
    import io
    import urllib.error

    calls = []

    def flaky(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))  # the edge's other throttle
        if len(calls) == 2:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(weather.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    assert weather._get_json("https://x", {"a": 1}) == {"ok": True} and len(calls) == 3
    # Open-Meteo throttles anonymous clients (429 for Python-urllib, resets for
    # generic agents): every request, retries included, identifies this project.
    assert calls[0].full_url == "https://x?a=1"
    assert all(c.get_header("User-agent") == weather.USER_AGENT for c in calls)

    def always_404(url, timeout):
        raise urllib.error.HTTPError(url, 404, "nope", {}, None)

    monkeypatch.setattr(weather.urllib.request, "urlopen", always_404)
    with pytest.raises(urllib.error.HTTPError):
        weather._get_json("https://x", {})

    resets = []

    def always_reset(req, timeout):
        resets.append(req)
        raise TimeoutError("The handshake operation timed out")

    monkeypatch.setattr(weather.urllib.request, "urlopen", always_reset)
    with pytest.raises(TimeoutError):
        weather._get_json("https://x", {})
    assert len(resets) == 3  # two retries, then the error surfaces
