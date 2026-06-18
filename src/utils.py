def get_severity_meta(severity: str) -> dict:
    """
    Returns a dict with css_class, dot, and label for the given severity string.
    Used by app.py to render the correct badge style.
    """
    s = severity.strip().lower()
    if s in ("life-threatening", "critical"):
        return {"css": "sev-critical", "dot": "🔴", "call": True}
    elif s == "severe":
        return {"css": "sev-critical", "dot": "🔴", "call": True}
    elif s == "moderate to severe":
        return {"css": "sev-severe", "dot": "🟠", "call": False}
    elif s == "high":
        return {"css": "sev-severe", "dot": "🟠", "call": False}
    elif s == "moderate":
        return {"css": "sev-moderate", "dot": "🟡", "call": False}
    elif s in ("minor", "minor to moderate"):
        return {"css": "sev-moderate", "dot": "🟢", "call": False}
    else:
        return {"css": "sev-moderate", "dot": "⚪", "call": False}


def get_severity_icon(severity: str) -> str:
    """Legacy single-string version kept for backward compatibility."""
    s = severity.strip().lower()
    if s in ("life-threatening",):
        return "🚨 LIFE-THREATENING — Call Emergency Services Immediately"
    elif s == "critical":
        return "🚨 CRITICAL — Call Emergency Services Immediately"
    elif s == "severe":
        return "🔴 SEVERE — Seek Immediate Medical Attention"
    elif s == "moderate to severe":
        return "🔴 MODERATE TO SEVERE — Seek Medical Attention Promptly"
    elif s == "high":
        return "🔴 HIGH RISK"
    elif s == "moderate":
        return "🟡 MODERATE — Monitor and Treat; See a Doctor if Worsening"
    elif s in ("minor", "minor to moderate"):
        return "🟢 MINOR — Can Usually Be Managed at Home"
    else:
        return f"⚠️ {severity.upper()} — Please assess the situation carefully"
