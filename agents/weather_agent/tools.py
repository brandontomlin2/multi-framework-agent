"""Tools for the example weather agent.

A tool is a plain, typed Python function with a Google-style docstring. The
``@tool`` decorator turns it into a portable ``ToolSpec``; the same function
runs inside LangGraph (local, Agent Runtime, Cloud Run) and as a Claude
Managed Agents custom tool.
"""

from __future__ import annotations

from mfa import tool

from agents.weather_agent import weather


@tool
def get_weather_today(location: str) -> dict:
    """Get today's weather for a place: current conditions, today's high and low, and chance of precipitation.

    Temperatures are in both Celsius and Fahrenheit. Only call this once the user has said where.

    Args:
        location: City or place name, optionally with region or country, e.g. "Paris" or "Portland, Maine".
    """
    return weather.today(location)


TOOLS = [get_weather_today]
