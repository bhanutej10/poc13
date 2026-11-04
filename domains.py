"""Business domains (functions) a question can belong to, plus a keyword fallback.

The LLM returns the domain together with the intent. The domain follows the SUBJECT of the question;
the intent follows the TYPE of request ("What is the refund policy?" and "How many refunds were issued?"
are both Finance, but DOC_RETRIEVAL and STRUCTURED_DATA). Anything that fits no function is "General".
"""
from __future__ import annotations

import re

GENERAL = "General"

DOMAINS = [
    "Race Operations",
    "Registration and Athlete Services",
    "Finance",
    "IT",
    "HR",
    "Legal and Compliance",
    "Procurement and Vendors",
    "Marketing and Sponsorship",
    "Medical and Safety",
    "Facilities and Logistics",
    "Volunteers and Staff Operations",
    "Travel and Accommodation",
    GENERAL,
]

_ALIASES = {
    "race ops": "Race Operations", "race operations": "Race Operations", "operations": "Race Operations",
    "registration": "Registration and Athlete Services", "athlete services": "Registration and Athlete Services",
    "registration and athlete services": "Registration and Athlete Services",
    "finance": "Finance", "accounting": "Finance", "billing": "Finance", "payments": "Finance",
    "it": "IT", "information technology": "IT", "tech": "IT", "technology": "IT",
    "hr": "HR", "human resources": "HR",
    "legal": "Legal and Compliance", "compliance": "Legal and Compliance", "legal and compliance": "Legal and Compliance",
    "procurement": "Procurement and Vendors", "vendors": "Procurement and Vendors",
    "procurement and vendors": "Procurement and Vendors",
    "marketing": "Marketing and Sponsorship", "sponsorship": "Marketing and Sponsorship",
    "marketing and sponsorship": "Marketing and Sponsorship",
    "medical": "Medical and Safety", "safety": "Medical and Safety", "medical and safety": "Medical and Safety",
    "facilities": "Facilities and Logistics", "logistics": "Facilities and Logistics",
    "facilities and logistics": "Facilities and Logistics",
    "volunteers": "Volunteers and Staff Operations", "staff operations": "Volunteers and Staff Operations",
    "volunteers and staff operations": "Volunteers and Staff Operations",
    "travel": "Travel and Accommodation", "accommodation": "Travel and Accommodation",
    "travel and accommodation": "Travel and Accommodation",
    "general": GENERAL, "other": GENERAL, "none": GENERAL, "unknown": GENERAL, "n/a": GENERAL, "": GENERAL,
}
_CANON = {d.lower(): d for d in DOMAINS}


def normalize_domain(v) -> str:
    """Map whatever the model returned onto one of DOMAINS; unknown values become General."""
    s = re.sub(r"[\s_\-/]+", " ", str(v or "").strip()).lower()
    return _CANON.get(s) or _ALIASES.get(s, GENERAL)


# Keyword fallback, used only when the LLM is unavailable. Order = tie-break priority. Deliberately
# narrow: an unknown domain is reported as General rather than guessed.
_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("Medical and Safety", re.compile(
        r"\b(medical|medic|injur\w*|collaps\w*|unconscious|ambulance|heat (?:stroke|illness)|first aid|"
        r"safety|fire|incident|doctor|physio\w*|ibuprofen)\b", re.I)),
    ("Legal and Compliance", re.compile(
        r"\b(waiver|lawyer|legal|lawsuit|sue|cease and desist|gdpr|data protection|privacy|compliance|"
        r"nda|non-disclosure|data retention)\b", re.I)),
    ("Finance", re.compile(
        r"\b(invoices?|refunds?|payments?|billed|billing|charged?|expenses?|reimburse\w*|revenue|budget|"
        r"wire|exchange rate|salary slip|credit card|corporate card|unpaid)\b", re.I)),
    ("IT", re.compile(
        r"\b(vpn|laptop|wifi|wi-fi|password|phishing|hacked|software|licen[cs]e|server|network|"
        r"login|it ticket|helpdesk|help desk|timing system)\b", re.I)),
    ("HR", re.compile(
        r"\b(leave|payroll|salary|employee|employees|new hires?|onboarding|parental|dress code|"
        r"harass\w*|discriminat\w*|manager keeps)\b", re.I)),
    ("Procurement and Vendors", re.compile(
        r"\b(vendors?|suppliers?|purchase orders?|\bpo\b|bids?|procurement|kickbacks?)\b", re.I)),
    ("Marketing and Sponsorship", re.compile(
        r"\b(sponsors?|sponsorship|brand|logo|campaign|instagram|twitter|social media|followers)\b", re.I)),
    ("Volunteers and Staff Operations", re.compile(
        r"\b(volunteers?|shifts?|briefing|aid station volunteers|meal vouchers?)\b", re.I)),
    ("Travel and Accommodation", re.compile(
        r"\b(hotels?|flights?|visa|accommodation|airport|travel)\b", re.I)),
    ("Facilities and Logistics", re.compile(
        r"\b(venue|parking|shipments?|shipping|containers?|warehouse|expo hall|bike transport|"
        r"bike racks?|barriers|finish arch)\b", re.I)),
    ("Registration and Athlete Services", re.compile(
        r"\b(registrations?|registered|register|entry|entries|check-?in|deferral|athlete id|waitlist)\b", re.I)),
    ("Race Operations", re.compile(
        r"\b(swim|bike leg|run course|cut-?off|drafting|marshal|splits?|finish line|aid stations?|"
        r"transition|dnfs?|wave|course|race day|finishers?)\b", re.I)),
]


def detect_domain(query: str) -> str:
    for name, rx in _KEYWORDS:
        if rx.search(query or ""):
            return name
    return GENERAL
