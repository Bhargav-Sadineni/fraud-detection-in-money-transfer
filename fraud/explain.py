"""Plain-English analyst notes. Uses Claude if ANTHROPIC_API_KEY is set, else a template."""
import os
import httpx


def template_note(sender, receiver, amount, status, reasons) -> str:
    if status == "COMPLETED":
        return ""
    why = "; ".join(reasons) if reasons else "the combined risk score was high"
    verb = "blocked" if status == "BLOCKED" else "held for review"
    return f"Transfer of ${amount:,.2f} from {sender} to {receiver} was {verb} because: {why}."


def llm_note(facts: str):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=15, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"), "max_tokens": 200,
                  "messages": [{"role": "user", "content":
                      "You assist a bank fraud analyst. In 2-3 plain sentences, explain why this transfer "
                      "was flagged and what the analyst should verify before approving. Do not invent facts.\n\n" + facts}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip() or None
    except Exception:
        return None
