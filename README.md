# NCA Volunteer Matching & Communication Assistant

**Volunteer Engagement Department — Northbridge Community Alliance**  
GBA 479 Take Home Final · Simon Business School · 2026

---

## What It Does

An intelligent volunteer coordination assistant for NCA program managers. Describe a volunteer need in plain language — the system applies deterministic certification rules, availability filters, and a fairness-aware scoring algorithm to return transparent match results with full explanations. No match found? The system explains exactly which constraint eliminated each candidate and suggests what to adjust.

---

## Architecture

```
Coordinator Request (plain language)
     │
     ▼
┌─────────────────────────────────┐
│  PARSER  [LLM — Haiku]          │  Extracts structured criteria:
│                                 │  role, days, time, notice, hours,
└──────────────┬──────────────────┘  transport, language, skills
               │
               ▼
┌─────────────────────────────────┐
│  MATCHING ENGINE  [Python]      │  Hard rules (cannot be overridden):
│  match_volunteers()             │  ① Certification check
│                                 │  ② Availability (day + time block)
│  cert_check()                   │  ③ Notice period
│  availability_check()           │  ④ Weekly hours limit
│  skill_score()                  │  ⑤ Transport requirement
│  recency_score()                │  ⑥ Language requirement
└──────────────┬──────────────────┘  Then: skill + fairness scoring
               │
               ▼
┌─────────────────────────────────┐
│  SUMMARY  [LLM — Opus]          │  Explains matches + drafts
│                                 │  assignment confirmation using
└──────────────┬──────────────────┘  NCA standard onboarding language
               │
               ▼
┌─────────────────────────────────┐
│  VALIDATOR  [Python]            │  Cert integrity check — verifies
│  validate_match_result()        │  matched volunteers actually hold
└─────────────────────────────────┘  required certs; no-match path check
```

**Task decomposition:**
- **Python (deterministic):** All filtering, certification enforcement, scoring, and validation
- **LLM — Haiku:** Natural language → structured criteria extraction
- **LLM — Opus:** Match explanation and confirmation drafting

---

## Certification Hard Rules

These are Python code — the LLM cannot override them under any circumstances.

| Role Type | Required Certifications |
|-----------|------------------------|
| Youth-facing (tutoring, mentoring) | Background Check — Cleared **+** Child Safety Training — Completed |
| Pantry food handling | Food Safety — Basic |
| Driving / Delivery | Driver Authorization — Approved |
| Environmental, Events | No certification required |

**Pending certifications:** Volunteers with pending certs are not matched for independent roles. They appear in the rejection summary with reason "Cert pending/missing."

---

## Matching & Scoring

Candidates who pass all hard rules are scored:

```
Total Score = (Skill Score × 0.7) + (Recency Score × 0.3)

Skill Score (0–10):
  +3 per preferred role alignment
  +2 per direct skill match
  +2 per required skill match
  +1 language bonus (if bilingual role)

Recency Score (0–10):
  Higher score = longer since last assignment
  → Volunteers not recently assigned get priority
  → Prevents the same people being asked repeatedly
```

---

## Key Design Decisions

**Certification rules are code, not prompts.** Youth roles require Background Check + Child Safety Training. Pantry food handling requires Food Safety cert. These are hard Python conditions — the LLM cannot reason around them or make exceptions.

**Fairness-aware scoring distributes load.** 30% of the match score is based on how long it has been since a volunteer was last assigned. Volunteers with no recent assignments get higher priority, preventing chronic over-reliance on the same individuals.

**Transparent rejection paths.** Every excluded volunteer gets a named reason (e.g. "Not available on Sat", "Missing required cert: Food Safety - Basic", "Requires 5+ days notice"). Managers see exactly why someone wasn't matched — not just who was.

**No-match escalation.** If no volunteers qualify, the system explains the binding constraint(s) and suggests concrete adjustments (change timing, contact a pending-cert volunteer, post a general call).

---

## Data

| File | Content |
|------|---------|
| `northbridge_volunteer_roster.csv` | 20 volunteers — skills, certifications, availability, languages, transport, notice requirements, max hours |

---

## Setup

```bash
pip install anthropic pandas
export ANTHROPIC_API_KEY=your_key_here
python nca_volunteer_assistant.py
# or specify CSV path:
python nca_volunteer_assistant.py path/to/northbridge_volunteer_roster.csv
```

---

## Example Interactions

```
You: I need a tutor for Monday evening, youth math support

→ Parser: role=youth tutoring math, days=[Mon], time=[Evening], notice=3
→ Required certs: Background Check - Cleared + Child Safety Training - Completed
→ Matched: 1 (Aisha — only active volunteer with both certs + Mon evening availability)
→ Score: skill=7, recency=3.1, total=5.83

Draft confirmation:
  "Hi Aisha, thank you for volunteering with Northbridge.
   This message confirms your tutoring assignment on Monday evening..."
```

```
You: Pantry shift Thursday evening, bilingual Spanish preferred

→ Required certs: Food Safety - Basic
→ Filters: Thu + Evening + Spanish language
→ Matched candidates with score breakdown + transparent rejection reasons
```

```
You: Driver needed Saturday morning, 3-hour delivery run

→ Required certs: Driver Authorization - Approved + transport=Car
→ If no match: explains which volunteers came closest and what cert is missing
→ Suggests: "Contact V-0006 who has Driver Authorization — Pending;
   approval may resolve this constraint"
```

---

## Governance Dimensions

| Dimension | Implementation |
|-----------|---------------|
| Risk Classification | Hard cert rules by role type |
| Auditability | Full rejection log with per-volunteer reasons |
| Fairness & Bias | Recency-weighted scoring prevents assignment concentration |
| Human Oversight | No-match path requires manager decision |
| Transparency | Score breakdown exposed for every candidate |
