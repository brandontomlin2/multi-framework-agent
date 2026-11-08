---
name: weather-briefing
description: Turn today's weather for a place into practical advice - what to wear, whether to carry an umbrella, and a short briefing for the day. Use this whenever the user asks what to wear, how to dress, what to pack, whether they need an umbrella, a coat or a jacket, or asks for a briefing or a plan for their day, even if they mention the weather only in passing. Needs today's weather from get_weather_today first.
license: Apache-2.0
metadata:
  author: multi-framework-agent
  version: "1.0"
---

# Weather briefing

Use this after you have today's weather for the place. Call `get_weather_today`
first; if the user has not said where, ask before doing anything else.

## Steps

1. Read `references/clothing-guide.md` (in this skill's folder) and pick:
   - the clothing band that contains today's **high** in °C,
   - the rain rule for today's **chance of precipitation**,
   - the wind note if the current wind is 20 km/h or more.
2. Reply in exactly this shape, as plain text with these three labelled lines
   and nothing else before them:

   ```
   Briefing for <Place, Region or Country>: <conditions>, <current>°F (<current>°C), high <high>°F (<high>°C), low <low>°F (<low>°C), <chance>% chance of rain.
   Wear: <the clothing band's advice, one sentence>.
   Rain: <the rain rule, one sentence>.
   ```

   Add a fourth line `Wind: ...` only when the wind note applies.
3. Every number comes from the tool result. Never estimate or round beyond
   what the tool returned. Use the guide's wording for the clothing and rain
   lines so advice is consistent from day to day.

If the user asked only one of the questions (for example just about an
umbrella), still give the full briefing; it is short.
