"""Risk assessment: combines detected allergens with the user's allergies.

Risk levels:
  HIGH   - the allergen is a direct ingredient
  MEDIUM - the allergen appears only in a 'may contain / traces' warning
  LOW    - the allergen was not detected
"""

import re
import sys

from src.detector import detect_allergens

# Text after phrases like "may contain" is treated as a trace warning.
TRACE_PATTERN = re.compile(
    r"(may contain|traces? of|produced in a facility|"
    r"manufactured in a factory)(.*?)(?:storage|best before|\.|$)",
    re.IGNORECASE | re.DOTALL,
)

LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def split_text(text):
    """Split label text into (main ingredient text, trace warning text)."""
    trace_parts = []

    def grab(match):
        trace_parts.append(match.group(2))
        return " "

    main = TRACE_PATTERN.sub(grab, text)
    return main, " ".join(trace_parts)


def match_user_allergies(user_keywords):
    """Map user keywords (e.g. 'milk') to allergen group names."""
    from src.detector import ALLERGENS

    selected = []
    for kw in user_keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        for name in ALLERGENS:
            if kw in name.lower() and name not in selected:
                selected.append(name)
    return selected


def assess_risk(text, user_allergies):
    """Assess the risk of a product for a user.

    Args:
        text: ingredient text (OCR output).
        user_allergies: list of allergen group names the user is allergic to.

    Returns:
        dict with 'overall', 'per_allergen', 'direct' and 'traces'.
    """
    main_text, trace_text = split_text(text)
    direct = detect_allergens(main_text)
    traces = detect_allergens(trace_text)

    per_allergen = {}
    for allergen in user_allergies:
        if allergen in direct:
            per_allergen[allergen] = "HIGH"
        elif allergen in traces:
            per_allergen[allergen] = "MEDIUM"
        else:
            per_allergen[allergen] = "LOW"

    overall = "LOW"
    for level in per_allergen.values():
        if LEVEL_ORDER[level] > LEVEL_ORDER[overall]:
            overall = level

    return {
        "overall": overall,
        "per_allergen": per_allergen,
        "direct": direct,
        "traces": traces,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python -m src.risk <image_path> "milk,soy"')
        sys.exit(1)

    from src.ocr import extract_text

    text = extract_text(sys.argv[1])
    allergies = match_user_allergies(sys.argv[2].split(","))
    print("User allergies:", ", ".join(allergies) or "none matched")

    result = assess_risk(text, allergies)
    print("\nOVERALL RISK:", result["overall"])
    for name, level in result["per_allergen"].items():
        print(f"  {name}: {level}")
    print("\nDirect ingredients:", ", ".join(result["direct"]) or "none")
    print("Trace warnings:", ", ".join(result["traces"]) or "none")