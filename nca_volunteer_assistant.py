"""
NCA Volunteer Engagement — Matching & Communication Assistant
GBA 479 — Volunteer Engagement Department
Northbridge Community Alliance

Architecture:
  - Roster loader      (Python/deterministic): reads volunteer CSV
  - Matching engine    (Python/deterministic): filters by cert, availability,
                                               skills, notice, hours — hard rules
  - Ranker             (Python/deterministic): scores candidates by fit + fairness
  - Interpreter        (LLM/Claude Haiku):     parses natural language request → structured need
  - Summary generator  (LLM/Claude Opus):      explains match results + drafts confirmation
  - Validator          (Python/deterministic): checks cert rules + no-match escalation path

Certification hard rules (from manual):
  - Youth-facing roles → Background Check - Cleared + Child Safety Training - Completed
  - Pantry food handling → Food Safety - Basic
  - Driving/Delivery → Driver Authorization - Approved
  - Pending certs → shadow/orientation only, no independent assignment
"""

import os
import json
import re
import pandas as pd
from typing import Optional
from anthropic import Anthropic

# ─────────────────────────────────────────────
# ROSTER LOADER  (deterministic)
# ─────────────────────────────────────────────

def load_roster(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse semicolon-separated list fields into Python lists
    list_cols = ["skills", "preferred_roles", "certifications",
                 "availability_days", "availability_time_blocks", "languages"]
    for col in list_cols:
        df[col] = df[col].fillna("").apply(
            lambda x: [v.strip() for v in x.split(";") if v.strip()]
        )
    df["min_notice_days"] = pd.to_numeric(df["min_notice_days"], errors="coerce").fillna(0)
    df["max_hours_per_week"] = pd.to_numeric(df["max_hours_per_week"], errors="coerce").fillna(0)
    return df


# ─────────────────────────────────────────────
# CERTIFICATION RULES  (deterministic hard rules)
# ─────────────────────────────────────────────

CERT_RULES = {
    "youth": ["Background Check - Cleared", "Child Safety Training - Completed"],
    "tutoring": ["Background Check - Cleared", "Child Safety Training - Completed"],
    "pantry": ["Food Safety - Basic"],
    "food pantry": ["Food Safety - Basic"],
    "driving": ["Driver Authorization - Approved"],
    "delivery": ["Driver Authorization - Approved"],
}

YOUTH_ROLE_KEYWORDS = ["tutor", "tutoring", "youth", "child", "mentor", "kids", "school", "student"]
PANTRY_ROLE_KEYWORDS = ["pantry", "food", "sorting", "stocking", "distribution", "intake"]
DRIVING_ROLE_KEYWORDS = ["driv", "deliver", "transport"]


def required_certs_for_role(role_description: str) -> list[str]:
    """Deterministically infer required certifications from role text."""
    text = role_description.lower()
    required = []
    if any(kw in text for kw in YOUTH_ROLE_KEYWORDS):
        required += CERT_RULES["youth"]
    if any(kw in text for kw in PANTRY_ROLE_KEYWORDS):
        required += CERT_RULES["pantry"]
    if any(kw in text for kw in DRIVING_ROLE_KEYWORDS):
        required += CERT_RULES["driving"]
    return list(set(required))


def has_cert(volunteer: pd.Series, cert: str) -> bool:
    return cert in volunteer["certifications"]


def cert_check(volunteer: pd.Series, required: list[str]) -> tuple[bool, list[str]]:
    """Returns (passes, missing_certs)."""
    missing = [c for c in required if not has_cert(volunteer, c)]
    return len(missing) == 0, missing


# ─────────────────────────────────────────────
# AVAILABILITY MATCHING  (deterministic)
# ─────────────────────────────────────────────

DAY_MAP = {
    "monday": "Mon", "mon": "Mon",
    "tuesday": "Tue", "tue": "Tue",
    "wednesday": "Wed", "wed": "Wed",
    "thursday": "Thu", "thu": "Thu",
    "friday": "Fri", "fri": "Fri",
    "saturday": "Sat", "sat": "Sat",
    "sunday": "Sun", "sun": "Sun",
}

TIME_MAP = {
    "morning": "Morning", "am": "Morning",
    "midday": "Midday", "noon": "Midday", "lunch": "Midday",
    "afternoon": "Afternoon", "pm": "Afternoon",
    "evening": "Evening", "night": "Evening",
}


def normalize_days(days: list[str]) -> list[str]:
    return [DAY_MAP.get(d.lower(), d) for d in days]


def normalize_time(time_blocks: list[str]) -> list[str]:
    return [TIME_MAP.get(t.lower(), t) for t in time_blocks]


def availability_check(volunteer: pd.Series,
                       required_days: list[str],
                       required_time: list[str],
                       notice_days: int,
                       shift_hours: float) -> tuple[bool, str]:
    """Returns (passes, reason_if_fails)."""
    norm_days = normalize_days(required_days)
    norm_time = normalize_time(required_time)

    # Day check
    if norm_days and not any(d in volunteer["availability_days"] for d in norm_days):
        return False, f"Not available on {'/'.join(norm_days)}"

    # Time check
    if norm_time and not any(t in volunteer["availability_time_blocks"] for t in norm_time):
        return False, f"Not available during {'/'.join(norm_time)}"

    # Notice check
    if notice_days < volunteer["min_notice_days"]:
        return False, f"Requires {int(volunteer['min_notice_days'])}+ days notice"

    # Hours check
    if shift_hours > volunteer["max_hours_per_week"]:
        return False, f"Shift ({shift_hours}h) exceeds weekly max ({int(volunteer['max_hours_per_week'])}h)"

    return True, ""


# ─────────────────────────────────────────────
# SKILL / ROLE MATCHING  (deterministic scoring)
# ─────────────────────────────────────────────

def skill_score(volunteer: pd.Series, role_description: str, required_skills: list[str]) -> int:
    """Score 0–10 based on skill and role alignment."""
    text = role_description.lower()
    score = 0
    # Preferred role alignment
    for pr in volunteer["preferred_roles"]:
        if pr.lower() in text or any(kw in pr.lower() for kw in text.split()):
            score += 3
    # Direct skill match
    for skill in volunteer["skills"]:
        if skill.lower() in text:
            score += 2
    # Required skills match
    for rs in required_skills:
        if rs in volunteer["skills"] or rs in volunteer["preferred_roles"]:
            score += 2
    # Language bonus (capped at 1)
    lang_keywords = ["bilingual", "spanish", "arabic", "french", "urdu", "korean",
                     "vietnamese", "swahili", "bengali", "russian", "tamil", "twi", "japanese"]
    if any(lk in text for lk in lang_keywords):
        for lang in volunteer["languages"]:
            if lang.lower() in text:
                score += 1
                break
    return min(score, 10)


def recency_score(volunteer: pd.Series) -> float:
    """Lower score = assigned more recently = lower priority (fair distribution)."""
    last = volunteer.get("last_assigned_date", "")
    if pd.isna(last) or last == "":
        return 10.0  # Never assigned → highest priority
    try:
        last_dt = pd.to_datetime(last, format="%m/%d/%Y", errors="coerce")
        if pd.isna(last_dt):
            return 5.0
        days_since = (pd.Timestamp.now() - last_dt).days
        return min(days_since / 30.0, 10.0)  # normalize to 0-10
    except Exception:
        return 5.0


# ─────────────────────────────────────────────
# CORE MATCHING ENGINE  (deterministic)
# ─────────────────────────────────────────────

def match_volunteers(df: pd.DataFrame,
                     role_description: str,
                     required_skills: list[str],
                     required_days: list[str],
                     required_time: list[str],
                     notice_days: int,
                     shift_hours: float,
                     require_transport: bool = False,
                     required_language: Optional[str] = None,
                     top_k: int = 3) -> dict:
    """
    Deterministic matching pipeline.
    Returns matched candidates + rejection reasons for transparency.
    """
    required_certs = required_certs_for_role(role_description)

    matched = []
    rejected = []

    for _, vol in df.iterrows():
        if vol["status"] not in ("Active", "Onboarding"):
            continue

        # Hard rule: certification check
        cert_ok, missing_certs = cert_check(vol, required_certs)
        if not cert_ok:
            # Pending check — shadow only
            pending = any("Pending" in c for c in vol["certifications"])
            if pending and missing_certs:
                rejected.append({
                    "id": vol["volunteer_id"],
                    "name": vol.get("preferred_name") or vol["first_name"],
                    "reason": f"Cert pending/missing: {', '.join(missing_certs)}"
                })
            else:
                rejected.append({
                    "id": vol["volunteer_id"],
                    "name": vol.get("preferred_name") or vol["first_name"],
                    "reason": f"Missing required cert(s): {', '.join(missing_certs)}"
                })
            continue

        # Hard rule: availability check
        avail_ok, avail_reason = availability_check(
            vol, required_days, required_time, notice_days, shift_hours
        )
        if not avail_ok:
            rejected.append({
                "id": vol["volunteer_id"],
                "name": vol.get("preferred_name") or vol["first_name"],
                "reason": avail_reason
            })
            continue

        # Hard rule: transport
        if require_transport and vol["transportation"] != "Car":
            rejected.append({
                "id": vol["volunteer_id"],
                "name": vol.get("preferred_name") or vol["first_name"],
                "reason": "No personal vehicle (required for this role)"
            })
            continue

        # Hard rule: language
        if required_language and required_language not in vol["languages"]:
            rejected.append({
                "id": vol["volunteer_id"],
                "name": vol.get("preferred_name") or vol["first_name"],
                "reason": f"Does not speak {required_language}"
            })
            continue

        # Scoring
        s_skill = skill_score(vol, role_description, required_skills)
        s_recency = recency_score(vol)
        total_score = s_skill * 0.7 + s_recency * 0.3

        matched.append({
            "volunteer_id": vol["volunteer_id"],
            "name": vol.get("preferred_name") or vol["first_name"],
            "pronouns": vol.get("pronouns", ""),
            "skills": vol["skills"],
            "preferred_roles": vol["preferred_roles"],
            "certifications": vol["certifications"],
            "availability_days": vol["availability_days"],
            "availability_time_blocks": vol["availability_time_blocks"],
            "languages": vol["languages"],
            "transportation": vol["transportation"],
            "max_hours_per_week": int(vol["max_hours_per_week"]),
            "notes": vol.get("notes", ""),
            "last_assigned_date": vol.get("last_assigned_date", "Never"),
            "contact": vol.get("preferred_contact", "Email"),
            "skill_score": s_skill,
            "recency_score": round(s_recency, 1),
            "total_score": round(total_score, 2),
        })

    matched.sort(key=lambda x: -x["total_score"])
    top_matches = matched[:top_k]

    return {
        "ok": True,
        "role_description": role_description,
        "required_certs": required_certs,
        "total_candidates": len(df),
        "matched_count": len(matched),
        "top_matches": top_matches,
        "rejected_count": len(rejected),
        "rejection_summary": rejected[:8],
        "no_match": len(matched) == 0,
    }


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

PARSER_SYSTEM = """You are a request parser for the NCA Volunteer Matching system.
Extract structured matching criteria from a natural language coordinator request.

Return ONLY a JSON object (no markdown):
{
  "role_description": "<full role description as given>",
  "required_skills": ["<skill1>", "<skill2>"],
  "required_days": ["<Mon|Tue|Wed|Thu|Fri|Sat|Sun>"],
  "required_time": ["<Morning|Midday|Afternoon|Evening>"],
  "notice_days": <integer — how many days notice is available, default 3>,
  "shift_hours": <float — estimated shift length in hours, default 2.5>,
  "require_transport": <true|false>,
  "required_language": "<language or null>",
  "top_k": <integer — how many matches to return, default 3>,
  "reasoning": "<one sentence>"
}

Known skills: Tutoring - Math, Tutoring - Science, Youth Mentoring, Pantry Operations,
Inventory/Sorting, Customer Service, Environmental Cleanup, Event Support, Data Entry,
Community Outreach, Driver, ESL Support, Intake/Translation, Photography/Media,
Adult Learning, Analytics/Reporting, Crafts/Activities, Forklift (experienced).

Default notice_days=3 if not specified. Default shift_hours=2.5 if not specified.
Set require_transport=true only if role explicitly requires driving or delivery.
"""

SUMMARY_SYSTEM = """You are the NCA Volunteer Coordination Assistant.
You explain volunteer match results to program managers and draft assignment confirmations.

Rules:
- Clearly explain WHY each top candidate was matched (skills, certs, availability).
- Use preferred names and correct pronouns.
- If no match found, explain the constraint(s) that eliminated all candidates and
  suggest what the manager can do (adjust timing, relax a constraint, contact pending cert volunteer).
- Draft a short confirmation message snippet for the top match using NCA standard language.
- Use person-first, professional language.
- Keep explanation under 250 words.
- End with: [Volunteer Coordination | Matched from NCA roster v{date}]
""".replace("{date}", "Feb 2026")

NO_MATCH_SYSTEM = """You are the NCA Volunteer Coordination Assistant.
No volunteers matched the given request.
Explain clearly:
1. What constraints eliminated candidates (cert missing, availability mismatch, etc.).
2. What options the manager has (adjust timing, contact pending cert volunteer, post a need).
Be specific about which volunteers came closest and what was missing.
Keep under 200 words.
End with: [Volunteer Coordination | No match found — escalation recommended]
"""


# ─────────────────────────────────────────────
# VALIDATOR  (deterministic)
# ─────────────────────────────────────────────

def validate_match_result(match_result: dict, response: str) -> tuple[bool, str]:
    """
    Checks:
    1. If match found: response mentions at least one volunteer name.
    2. If no match: response contains escalation note.
    3. Response contains coordination footer.
    4. Cert rules not violated (matched volunteers actually have required certs).
    """
    has_footer = "Volunteer Coordination" in response

    if match_result["no_match"]:
        has_escalation = "escalat" in response.lower() or "no match" in response.lower() or "adjust" in response.lower()
        if not has_escalation:
            return False, "No-match response missing escalation guidance."
    else:
        top = match_result["top_matches"]
        if top:
            name = top[0]["name"]
            if name not in response:
                return False, f"Response does not mention top match '{name}'."

        # Verify cert integrity — matched volunteers must have required certs
        required = match_result["required_certs"]
        for vol in match_result["top_matches"]:
            for cert in required:
                if cert not in vol["certifications"]:
                    return False, f"CERT VIOLATION: {vol['name']} matched but missing '{cert}'."

    if not has_footer:
        return False, "Response missing coordination footer."

    return True, "ok"


# ─────────────────────────────────────────────
# MAIN ASSISTANT
# ─────────────────────────────────────────────

def process_request(client: Anthropic, user_input: str, df: pd.DataFrame) -> dict:
    """Full pipeline: parse → match → summarize → validate."""

    # ── STEP 1: Parse natural language request (LLM — Haiku) ──
    parse_resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        system=PARSER_SYSTEM,
        messages=[{"role": "user", "content": user_input}],
    )
    criteria = json.loads(parse_resp.content[0].text.strip())

    # ── STEP 2: Match volunteers (Python — deterministic) ──
    match_result = match_volunteers(
        df=df,
        role_description=criteria["role_description"],
        required_skills=criteria.get("required_skills", []),
        required_days=criteria.get("required_days", []),
        required_time=criteria.get("required_time", []),
        notice_days=criteria.get("notice_days", 3),
        shift_hours=criteria.get("shift_hours", 2.5),
        require_transport=criteria.get("require_transport", False),
        required_language=criteria.get("required_language"),
        top_k=criteria.get("top_k", 3),
    )

    # ── STEP 3: Generate summary (LLM — Opus) ──
    system = NO_MATCH_SYSTEM if match_result["no_match"] else SUMMARY_SYSTEM
    gen_prompt = (
        f"Coordinator request: {user_input}\n\n"
        f"Matching criteria extracted: {json.dumps(criteria, indent=2)}\n\n"
        f"Match results: {json.dumps(match_result, indent=2)}\n\n"
        + ("Explain why no one matched and what options exist."
           if match_result["no_match"]
           else "Explain the top matches and draft a confirmation snippet for the top candidate.")
    )
    gen_resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": gen_prompt}],
    )
    response = gen_resp.content[0].text.strip()

    # ── STEP 4: Validate (Python — deterministic) ──
    passed, reason = validate_match_result(match_result, response)
    if not passed:
        # Regenerate with correction
        gen_resp2 = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            system=system,
            messages=[
                {"role": "user", "content": gen_prompt},
                {"role": "assistant", "content": response},
                {"role": "user", "content": f"Validation failed: {reason}. Please revise accordingly."},
            ],
        )
        response = gen_resp2.content[0].text.strip()
        passed, reason = validate_match_result(match_result, response)

    return {
        "criteria": criteria,
        "match_result": match_result,
        "response": response,
        "validation": {"passed": passed, "reason": reason},
    }


def display_result(result: dict):
    mr = result["match_result"]
    cr = result["criteria"]
    v_tag = "✅" if result["validation"]["passed"] else f"⚠️  {result['validation']['reason']}"

    print(f"\n{'═'*62}")
    print(f"  MATCH RESULTS — {cr['role_description'][:50]}")
    print(f"{'═'*62}")
    print(f"  Candidates screened : {mr['total_candidates']}")
    print(f"  Qualified matches   : {mr['matched_count']}")
    print(f"  Required certs      : {mr['required_certs'] or 'None'}")
    print(f"  Validation          : {v_tag}")

    if mr["top_matches"]:
        print(f"\n  TOP {len(mr['top_matches'])} MATCH(ES):\n")
        for i, v in enumerate(mr["top_matches"], 1):
            print(f"  {i}. {v['name']} ({v['pronouns']}) — {v['volunteer_id']}")
            print(f"     Skills     : {', '.join(v['skills'][:4])}")
            print(f"     Available  : {', '.join(v['availability_days'])} | {', '.join(v['availability_time_blocks'])}")
            print(f"     Languages  : {', '.join(v['languages'])}")
            print(f"     Transport  : {v['transportation']}")
            print(f"     Score      : skill={v['skill_score']} recency={v['recency_score']} total={v['total_score']}")
            if v["notes"]:
                print(f"     Notes      : {v['notes'][:80]}")
            print()
    else:
        print("\n  ⚠️  NO MATCHES FOUND\n")
        if mr["rejection_summary"]:
            print("  Elimination reasons:")
            for r in mr["rejection_summary"][:5]:
                print(f"    {r['id']} {r['name']}: {r['reason']}")
        print()

    print(f"\n  COORDINATOR SUMMARY:\n")
    for line in result["response"].split("\n"):
        print(f"  {line}")
    print()


def chat(df: pd.DataFrame):
    client = Anthropic()

    print("\n" + "═"*62)
    print("  NCA VOLUNTEER MATCHING & COMMUNICATION ASSISTANT")
    print("  Volunteer Engagement Department")
    print(f"  {len(df[df['status']=='Active'])} active volunteers | "
          f"{len(df[df['status']=='Onboarding'])} onboarding")
    print("═"*62)
    print("  Describe your volunteer need in plain language.")
    print("  Examples:")
    print("    'I need 2 tutors for Saturday morning, youth math support'")
    print("    'Pantry shift Thursday evening, bilingual Spanish preferred'")
    print("    'Driver needed Saturday, 3-hour delivery run'")
    print("  Type 'roster' to see all volunteers, 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nAssistant: Goodbye.")
            break

        if user_input.lower() == "roster":
            print(f"\n  {'ID':<8} {'Name':<15} {'Status':<12} {'Days':<25} {'Skills (first 2)'}")
            print("  " + "-"*80)
            for _, v in df.iterrows():
                days = ";".join(v["availability_days"][:4]) if isinstance(v["availability_days"], list) else v["availability_days"]
                skills = ", ".join(v["skills"][:2]) if isinstance(v["skills"], list) else v["skills"]
                print(f"  {v['volunteer_id']:<8} {(v.get('preferred_name') or v['first_name']):<15} "
                      f"{v['status']:<12} {days:<25} {skills}")
            print()
            continue

        try:
            result = process_request(client, user_input, df)
            display_result(result)
        except json.JSONDecodeError as e:
            print(f"\n  [Parse error: {e}. Please rephrase your request.]\n")
        except Exception as e:
            print(f"\n  [Error: {e}]\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    csv_path = "northbridge_volunteer_roster.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    try:
        df = load_roster(csv_path)
    except FileNotFoundError:
        try:
            df = load_roster("/mnt/user-data/uploads/northbridge_volunteer_roster.csv")
        except FileNotFoundError:
            print("Error: Could not find volunteer roster CSV.")
            sys.exit(1)

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    chat(df)
