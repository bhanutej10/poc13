"""Intent taxonomy shared by the classifier, router and evaluation."""
from enum import Enum


class Intent(str, Enum):
    DOC_RETRIEVAL = "DOC_RETRIEVAL"
    STRUCTURED_DATA = "STRUCTURED_DATA"
    CONVERSATION = "CONVERSATION"
    UNSUPPORTED = "UNSUPPORTED"
    ESCALATION = "ESCALATION"


INTENT_DEFINITIONS = {
    Intent.DOC_RETRIEVAL: (
        "Question answerable from unstructured documents: policies, handbooks, guides, "
        "manuals, FAQs, procedures, rules, 'how do I / what does the doc say' questions."
    ),
    Intent.STRUCTURED_DATA: (
        "Question that needs a query over records/tables: counts, lists, filters, "
        "aggregations, rankings, trends, or lookup of a specific record or ID."
    ),
    Intent.CONVERSATION: (
        "Greetings, thanks, goodbyes, acknowledgements, small talk, questions about the "
        "assistant itself, or follow-ups about the previous answer (repeat, rephrase, shorten)."
    ),
    Intent.UNSUPPORTED: (
        "Out of scope: general knowledge, weather, news, coding help, creative writing, "
        "medical/financial/legal/personal advice, product recommendations, or actions the system "
        "cannot perform (sending email, booking, buying, paying or transferring money, posting, "
        "approving, assigning, installing, deleting data, resetting passwords)."
    ),
    Intent.ESCALATION: (
        "Needs a human: explicit request for a person/manager/agent, formal complaints, "
        "legal threats, anger after repeated failures, billing disputes with no resolution, "
        "fraud or security incidents, or any emergency / safety / harassment report."
    ),
}

# When a query fits several intents, the earliest in this list wins.
PRIORITY = [
    Intent.ESCALATION,
    Intent.UNSUPPORTED,
    Intent.STRUCTURED_DATA,
    Intent.DOC_RETRIEVAL,
    Intent.CONVERSATION,
]
