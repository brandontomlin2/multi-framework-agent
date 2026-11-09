"""Print the Wear/Rain(/Wind) lines of a weather briefing from the clothing guide's tables.

    python3 briefing_lines.py --high-c 15 --precip 10 --wind-kmh 11

Standard library only, so it runs in any sandbox.
"""

import argparse

BANDS = [
    (25, "light, breathable clothes such as shorts and a t-shirt, plus sunscreen"),
    (18, "a t-shirt or light shirt, with a light layer for the evening if the low is under 15°C"),
    (10, "a light jacket or sweater over a long-sleeve shirt"),
    (3, "a warm coat and long trousers, with a hat or scarf if it is windy"),
]
FREEZING = "a heavy winter coat, hat, gloves and scarf"


def wear(high_c: float) -> str:
    for floor, advice in BANDS:
        if high_c >= floor:
            return advice
    return FREEZING


def rain(precip_pct: float) -> str:
    if precip_pct >= 60:
        return "bring an umbrella or a rain jacket; rain is likely"
    if precip_pct >= 30:
        return "an umbrella is worth having in your bag"
    return "no umbrella needed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--high-c", type=float, required=True)
    ap.add_argument("--precip", type=float, required=True)
    ap.add_argument("--wind-kmh", type=float, default=0.0)
    a = ap.parse_args()
    print(f"Wear: {wear(a.high_c)}.")
    print(f"Rain: {rain(a.precip)}.")
    if a.wind_kmh >= 20:
        print("Wind: it is breezy, so a windproof outer layer helps.")


if __name__ == "__main__":
    main()
