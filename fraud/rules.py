"""Behaviour rules that look at the user's history (things the ML model cannot see)."""


def rule_check(c: dict):
    pts, why = 0.0, []
    n = c["n_last_hour"]
    if n >= 6:
        pts += .85; why.append(f"{n} transfer attempts in the last hour")
    elif n >= 3:
        pts += .45; why.append(f"{n} transfer attempts in the last hour")
    if c["n_completed"] >= 3 and c["avg_amount"] > 0 and c["amount"] >= 5 * c["avg_amount"]:
        pts += .30; why.append(f"amount is {c['amount'] / c['avg_amount']:.0f}x this user's usual transfer")
    if c["new_receiver"] and c["amount"] >= .5 * c["balance"]:
        pts += .25; why.append("first transfer to this receiver and it is over half the balance")
    if c["receiver_age_h"] < 1:
        pts += .15; why.append("receiver account is less than 1 hour old")
    if c["hour"] <= 5 and c["amount"] >= .25 * c["balance"]:
        pts += .15; why.append("large transfer during night hours (UTC)")
    if c["blocked_24h"] >= 2:
        pts += .25; why.append("several blocked attempts in the last 24 hours")
    return round(min(1.0, pts), 3), why


def decide(ml: dict, rule_score: float) -> str:
    if ml["score"] >= ml["threshold"] or rule_score >= .8:
        return "BLOCKED"
    if ml["score"] >= ml["review_threshold"] or rule_score >= .4 or ml["anomaly"] >= .995:
        return "HELD"
    return "COMPLETED"
