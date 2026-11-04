# POC-013: Question Intent Classifier and Router

Takes a user question, decides what kind of request it is, and says where it should go next. It answers
three things for every question: the **intent**, the business **domain**, and the **action** to take.

| Intent | What the user wants | Action returned |
|---|---|---|
| `DOC_RETRIEVAL` | A rule, procedure or explanation from documents | `call_rag_pipeline` |
| `STRUCTURED_DATA` | Counts, lists, status or a lookup over records | `call_text_to_sql` |
| `CONVERSATION` | A greeting, thanks or a follow-up | `call_chat_llm` |
| `UNSUPPORTED` | Something out of scope, or an action the system cannot do | `reply_static` |
| `ESCALATION` | A person: a complaint, an emergency, a legal or fraud issue | `handoff_to_human` |

Domain is one of: Race Operations, Registration and Athlete Services, Finance, IT, HR, Legal and Compliance,
Procurement and Vendors, Marketing and Sponsorship, Medical and Safety, Facilities and Logistics,
Volunteers and Staff Operations, Travel and Accommodation, or General. It is reported with the result and does
not change the route.

## How it works

1. The question goes to the LLM (Gemini by default) with the intent definitions, a priority order and a few
   worked examples. The model returns JSON that is validated before use.
2. If two intents fit, the higher one wins: `ESCALATION > UNSUPPORTED > STRUCTURED_DATA > DOC_RETRIEVAL > CONVERSATION`.
3. The router turns the intent into an action. Below 0.6 confidence it asks the user to clarify instead,
   except for escalations, which always go to a person.
4. If the LLM call fails, a keyword-rules classifier answers instead and the result is marked `rules_fallback`.

Words that mean different things in different functions ("ticket", "account", "policy", "transfer") are classified
by what the user is asking for, not by the word itself. "How many tickets are open?" is a data question;
"Buy two tickets" is an action the system cannot perform.

### Repeated questions
Send a `session_id` and the first result for each question is remembered for that session. Asking the same
question again, later and even after other questions, returns that stored result with `"repeated": true`.
Rewordings that share keywords ("Tell me about the refund rules" for "What is the refund policy?") are matched too.
This runs locally, with no LLM call and no tokens. A stricter check stops lookalikes with different answers
from matching (different numbers, places, years or request types). Rewordings that share no keywords need the
optional embedding model, see [Paraphrase matching](#paraphrase-matching).

## Requirements

- Python 3.10 or newer
- A Gemini API key. The free tier works but is limited to 15 requests a minute.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q                          # expect: 42 passed
```

## Configuration

Set these as environment variables (PowerShell: `$env:NAME = "value"`).

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | none | Your Gemini key. Required for LLM runs. |
| `LLM_PROVIDER` | auto | `gemini`, `mistral`, `grok` or `anthropic`. Auto picks the first provider that has a key. |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model name. Models that Google has closed to new users return 404. |
| `LLM_MIN_INTERVAL` | `0` | Seconds between calls. Use `4.5` on the free tier. |
| `LLM_TIMEOUT` | `60` | Seconds to wait for one response. |
| `LLM_MAX_RETRIES` | `6` | Attempts on rate limits and timeouts. |
| `TESTSET_PATH` | `testset.csv` | Test set used for evaluation and for the few-shot examples. |
| `FEWSHOT_PER_INTENT` | `6` (`3` if the test set has no domain column) | Examples per intent placed in the prompt (DEV rows only). |
| `HISTORY_MAX` / `HISTORY_TTL` | `200` / `86400` | Questions remembered per session, and for how many seconds. |
| `SIMILARITY_BACKEND` | `auto` | `lexical`, `embedding`, `auto` or `off`. |
| `SIMILARITY_THRESHOLD` | per backend | Minimum similarity for a reworded repeat to match. |

Mistral and Grok use `MISTRAL_API_KEY` and `XAI_API_KEY` with `MISTRAL_MODEL` and `GROK_MODEL`.

## Run the evaluation

```bash
export GEMINI_API_KEY='your_key'
export GEMINI_MODEL=gemini-3.5-flash-lite LLM_PROVIDER=gemini LLM_MIN_INTERVAL=4.5
export TESTSET_PATH=ironman_testset_v2.xlsx

python evaluate.py --backend rules  --split all     # offline baseline, no API call
python evaluate.py --backend gemini --split dev     # tune here, as often as needed
python evaluate.py --backend gemini --split test    # final score: run once
```

The DEV rows are the only ones used as prompt examples, and TEST is held back for the final number. After one
TEST run with an LLM a lock file blocks another, because rerunning means tuning on TEST. A run where any
question fell back to rules prints a warning and does not count as valid.

Each run writes two files:

- `results_<backend>_<split>.xlsx`: every prediction with PASS/FAIL, per-intent metrics, confusion matrix,
  routing accuracy and accuracy by domain.
- `results_<backend>_<split>.md`: every wrong answer with its severity, a likely cause and a suggested fix.

The target is 90% accuracy. Domain accuracy and accuracy on the shared-word questions are reported alongside,
but they are not part of the pass mark.

## Run the API

```bash
uvicorn app:app --port 8000
```

Interactive docs are at `http://localhost:8000/docs`.

| Endpoint | Returns |
|---|---|
| `POST /classify` | Intent, domain, confidence, reason |
| `POST /route` | Route and action, plus the classification |
| `GET /health` | Status, backend in use, matcher in use |

```bash
curl -s -X POST localhost:8000/route -H "Content-Type: application/json" \
  -d '{"query": "I have not been paid for two months and need to speak to someone", "session_id": "demo"}'
```

```json
{
  "route": "escalation",
  "intent": "ESCALATION",
  "action": "handoff_to_human",
  "response": "I'm connecting you with a member of our team now. ...",
  "low_confidence": false,
  "domain": "Finance",
  "classification": { "intent": "ESCALATION", "confidence": 1, "reason": "...", "backend": "gemini", "domain": "Finance" },
  "repeated": false,
  "matched_question": null,
  "similarity": null
}
```

`response` is filled in only for escalations and unsupported questions. For the other intents it is `null`, because
the answer itself comes from the pipeline named in `action`. `backend` shows `rules_fallback` if the LLM call failed.

## Paraphrase matching

The default matcher compares words after removing filler and reducing words to their stems. On the 47 pairs in
`paraphrase_pairs.csv` it caught 18 of 18 rewordings that share keywords and wrongly matched none of 23 lookalikes.
It does not catch rewordings with different vocabulary ("How do I get my money back?" for "refund policy").

For those, install a small local embedding model (about 130 MB, downloaded once, no API calls):

```bash
pip install -r requirements-semantic.txt
export SIMILARITY_BACKEND=embedding
python check_similarity.py --backend embedding --sweep     # shows wrong matches at each threshold
export SIMILARITY_THRESHOLD=0.88                           # lowest value with zero wrong matches
```

A wrong match returns the wrong stored answer, while a missed one only costs one normal LLM call, so choose
the threshold on wrong matches first. Add your own pairs to `paraphrase_pairs.csv` before relying on it.

## Test set

`ironman_testset_v2.xlsx` (also `.csv`) holds 200 questions across 12 business functions, with 40 per intent and
20 cross-domain cases where one word has different meanings.

| Column | Meaning |
|---|---|
| `id` | Row id |
| `query` | The question |
| `label` | Correct intent |
| `domain` | Group the question was written for |
| `ambiguous_term` | Shared word in the question, if any |
| `expected_route` | Correct action |
| `expected_domain` | Correct domain, or `General` |

The DEV/TEST split (70 and 130 rows) and the difficulty flag are not in the visible sheet. They are stored in a hidden
`_splits` sheet in the workbook, or in `ironman_testset_v2.splits.csv` next to the CSV, and are matched by `id`.
To clean your own workbook the same way, run `python hide_columns.py your.xlsx`. The labels are written by hand,
so review them against your own definitions. The older 150-question `testset.xlsx` is a generic demo set.

## Project layout

| File | Role |
|---|---|
| `app.py` | FastAPI service |
| `classifier.py` | LLM backends, output schema, rules fallback, prompt |
| `router.py` | Intent to action, low-confidence handling |
| `intents.py`, `domains.py` | Intent and domain definitions |
| `history.py`, `similarity.py` | Session memory and reworded-repeat matching |
| `dataset.py`, `evaluate.py` | Test set loader and evaluation with error report |
| `build_testset_v2.py`, `testset_v2_data.py` | Source of the v2 test set |
| `check_similarity.py`, `paraphrase_pairs.csv` | Measure and tune paraphrase matching |
| `hide_columns.py` | Move split and difficulty out of a workbook's visible sheet |
| `list_gemini_models.py` | List the Gemini models your key can call |
| `test_api.py`, `test_pipeline.py` | Tests |


