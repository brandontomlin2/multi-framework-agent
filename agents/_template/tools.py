"""Replace with your agent's tools. Type hints + a Google-style docstring are required:
they become the JSON schema on every target."""

from mfa import tool


@tool
def echo(text: str) -> str:
    """Repeat text back to the user.

    Args:
        text: The text to repeat.
    """
    return text


TOOLS = [echo]
