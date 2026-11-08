"""Weather data providers for the example agent.

* ``open-meteo`` (default): live data from Open-Meteo's free geocoding and
  forecast APIs (no API key). Hosts: geocoding-api.open-meteo.com, api.open-meteo.com.
* ``fixture``: deterministic offline data for tests and offline eval runs.

Select with ``MFA_WEATHER_PROVIDER``. The provider is read on every call, so the
same deployed agent can be pointed at fixtures for a reproducible eval run.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Any

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 10

# WMO weather interpretation codes, as documented by Open-Meteo.
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def c_to_f(c: float | None) -> float | None:
    return None if c is None else round(c * 9 / 5 + 32, 1)


RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_DELAYS_S = (1.5, 3.0)  # Open-Meteo's free tier rate-limits bursts; two short retries ride them out


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt, delay in enumerate((*RETRY_DELAYS_S, None)):
        try:
            with urllib.request.urlopen(full, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUSES or delay is None:
                raise
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def _report(place: dict[str, Any], day: str, current: dict[str, Any], daily: dict[str, Any]) -> dict[str, Any]:
    """Shape shared by both providers, so the agent sees the same thing either way."""
    return {
        "found": True,
        "location": {k: place.get(k) for k in ("name", "admin1", "country", "latitude", "longitude", "timezone")},
        "date": day,
        "current": {
            "conditions": WMO_CODES.get(current.get("weather_code"), "unknown"),
            "temperature_c": current.get("temperature_c"),
            "temperature_f": c_to_f(current.get("temperature_c")),
            "feels_like_c": current.get("feels_like_c"),
            "feels_like_f": c_to_f(current.get("feels_like_c")),
            "wind_kmh": current.get("wind_kmh"),
        },
        "today": {
            "conditions": WMO_CODES.get(daily.get("weather_code"), "unknown"),
            "high_c": daily.get("high_c"),
            "high_f": c_to_f(daily.get("high_c")),
            "low_c": daily.get("low_c"),
            "low_f": c_to_f(daily.get("low_c")),
            "precipitation_chance_pct": daily.get("precip_pct"),
        },
    }


def _not_found(location: str) -> dict[str, Any]:
    return {"found": False, "query": location, "message": "No place matched that location."}


def open_meteo(location: str) -> dict[str, Any]:
    # Geocoding matches on the place name, so "Paris, France" is searched as "Paris"
    # and the rest is used to pick among the candidates.
    name, _, qualifier = location.partition(",")
    geo = _get_json(GEOCODE_URL, {"name": name.strip(), "count": 10, "language": "en", "format": "json"})
    results = geo.get("results") or []
    if not results:
        return _not_found(location)
    q = qualifier.strip().lower()
    place = next(
        (r for r in results if q and q in f"{r.get('admin1', '')} {r.get('country', '')} {r.get('country_code', '')}".lower()),
        results[0],
    )
    fc = _get_json(FORECAST_URL, {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": 1,
    })
    cur, daily = fc.get("current", {}), fc.get("daily", {})

    def first(key: str):
        values = daily.get(key) or [None]
        return values[0]

    report = _report(
        place,
        first("time") or date.today().isoformat(),
        {
            "weather_code": cur.get("weather_code"),
            "temperature_c": cur.get("temperature_2m"),
            "feels_like_c": cur.get("apparent_temperature"),
            "wind_kmh": cur.get("wind_speed_10m"),
        },
        {
            "weather_code": first("weather_code"),
            "high_c": first("temperature_2m_max"),
            "low_c": first("temperature_2m_min"),
            "precip_pct": first("precipitation_probability_max"),
        },
    )
    others = [r for r in results if r is not place and r.get("name", "").lower() == place.get("name", "").lower()]
    if others:
        report["other_matches"] = [f"{r.get('name')}, {r.get('admin1')}, {r.get('country')}" for r in others[:4]]
    return report


# --- offline fixtures ------------------------------------------------------

FIXTURES: dict[str, dict[str, Any]] = {
    "seattle": {
        "place": {"name": "Seattle", "admin1": "Washington", "country": "United States",
                  "latitude": 47.61, "longitude": -122.33, "timezone": "America/Los_Angeles"},
        "current": {"weather_code": 61, "temperature_c": 9.0, "feels_like_c": 6.5, "wind_kmh": 14.0},
        "daily": {"weather_code": 63, "high_c": 11.0, "low_c": 6.0, "precip_pct": 85},
    },
    "paris": {
        "place": {"name": "Paris", "admin1": "Île-de-France", "country": "France",
                  "latitude": 48.85, "longitude": 2.35, "timezone": "Europe/Paris"},
        "current": {"weather_code": 2, "temperature_c": 12.0, "feels_like_c": 10.5, "wind_kmh": 11.0},
        "daily": {"weather_code": 2, "high_c": 15.0, "low_c": 8.0, "precip_pct": 10},
    },
    "tokyo": {
        "place": {"name": "Tokyo", "admin1": "Tokyo", "country": "Japan",
                  "latitude": 35.69, "longitude": 139.69, "timezone": "Asia/Tokyo"},
        "current": {"weather_code": 0, "temperature_c": 18.0, "feels_like_c": 18.0, "wind_kmh": 6.0},
        "daily": {"weather_code": 1, "high_c": 20.0, "low_c": 12.0, "precip_pct": 0},
    },
    "london": {
        "place": {"name": "London", "admin1": "England", "country": "United Kingdom",
                  "latitude": 51.51, "longitude": -0.13, "timezone": "Europe/London"},
        "current": {"weather_code": 3, "temperature_c": 10.0, "feels_like_c": 8.0, "wind_kmh": 20.0},
        "daily": {"weather_code": 80, "high_c": 12.0, "low_c": 7.0, "precip_pct": 60},
    },
    "portland": {
        "place": {"name": "Portland", "admin1": "Oregon", "country": "United States",
                  "latitude": 45.52, "longitude": -122.68, "timezone": "America/Los_Angeles"},
        "current": {"weather_code": 3, "temperature_c": 8.0, "feels_like_c": 6.0, "wind_kmh": 9.0},
        "daily": {"weather_code": 61, "high_c": 10.0, "low_c": 5.0, "precip_pct": 70},
        "other_matches": ["Portland, Maine, United States", "Portland, Victoria, Australia"],
    },
}


def fixture(location: str) -> dict[str, Any]:
    key = location.split(",")[0].strip().lower()
    f = FIXTURES.get(key)
    if f is None:
        return _not_found(location)
    report = _report(f["place"], date.today().isoformat(), f["current"], f["daily"])
    if "other_matches" in f:
        report["other_matches"] = f["other_matches"]
    return report


def today(location: str) -> dict[str, Any]:
    provider = os.environ.get("MFA_WEATHER_PROVIDER", "open-meteo")
    if provider == "fixture":
        return fixture(location)
    try:
        return open_meteo(location)
    except Exception as exc:  # network/API failure: tell the agent, don't crash the turn
        return {"found": False, "query": location, "error": f"weather service unavailable: {exc}"}
