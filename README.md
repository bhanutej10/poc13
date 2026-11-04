# POC-013: Question Intent Classifier and Router

Classifies a question into one of five intents and routes it to a handler.

| Intent | Meaning | Route action |
|---|---|---|
| DOC_RETRIEVAL | Answerable from policies/guides | `call_rag_pipeline` |
| STRUCTURED_DATA | Needs a query over records | `call_text_to_sql` |
| CONVERSATION | Greetings, thanks, follow-ups | `call_chat_llm` |
| UNSUPPORTED | Out of scope / action not possible | `reply_static` |
| ESCALATION | Human wanted, complaint, emergency, legal | `handoff_to_human` |

Overlaps resolve by priority: ESCALATION > UNSUPPORTED > STRUCTURED_DATA > DOC_RETRIEVAL > CONVERSATION.
Confidence < 0.6 -> `clarify` (never for ESCALATION). Invalid or failed LLM output -> rules fallback (`rules_fallback`).

## Files
| File | Purpose |
|---|---|
| `intents.py` | Taxonomy, definitions, priority |
| `classifier.py` | LLM backends (Gemini, Mistral, Grok via OpenAI-compatible JSON; Anthropic optional), Pydantic `IntentResult` schema, rules baseline/fallback |
| `router.py` | Intent -> handler, low-confidence policy (handlers are stubs; `Router.register`) |
| `app.py` | FastAPI: `GET /health`, `POST /classify`, `POST /route` |
| `dataset.py` | Loads your .xlsx/.csv test set (flexible column names, DEV/TEST split) |
| `evaluate.py` | Runs classifier+router; scikit-learn metrics; routing check; PASS/FAIL; error-analysis report |
| `testset.xlsx/.csv`, `testset_data.py`, `build_testset.py` | Generic 150-query demo set (replace with your IRONMAN set) |
| `test_api.py`, `test_pipeline.py` | 42 pytest tests |
| `list_gemini_models.py` | Lists Gemini models your key can call |

## Workflow (matches the POC plan)
```
pip install -r requirements.txt
pytest -q                                              # 1. tests

# point everything at YOUR workbook (50 DEV / 75 TEST rows)
# Linux/Mac: export TESTSET_PATH=ironman_testset.xlsx   Windows PowerShell: $env:TESTSET_PATH="ironman_testset.xlsx"
# Linux/Mac: export GEMINI_API_KEY=...                  Windows PowerShell: $env:GEMINI_API_KEY="..."

python evaluate.py --backend rules  --split dev        # 2. offline smoke test
python evaluate.py --backend gemini --split dev        # 3. DEV: read results_gemini_dev.md, tune the prompt, repeat
python evaluate.py --backend gemini --split test       # 4. TEST: run ONCE for the 90% check (a lock file blocks reruns)

uvicorn app:app --reload                               # 5. API (set LLM_PROVIDER=gemini)
```
Outputs per run: `results_<backend>_<split>.xlsx` (predictions with PASS/FAIL, per-intent metrics, confusion matrix, routing)
and `results_<backend>_<split>.md` (every misclassification with severity, likely cause, suggested fix).

## Workbook format
Matched case-insensitively. Required: query (`query|question|text`) and intent (`intent|label|expected_intent`).
Optional: `split` (DEV/TEST), `expected_route`, `id`, `difficulty`. Labels like "Doc Retrieval" are normalised.

## Settings
`GEMINI_MODEL` (use `gemini-3.5-flash-lite`; 2.5 models are closed to new users), `MISTRAL_MODEL`, `GROK_MODEL`, `LLM_PROVIDER`,
`LLM_TIMEOUT` (default 60 s), `LLM_MIN_INTERVAL` (seconds between calls; `4.5` for Gemini free tier = 15 req/min),
`LLM_MAX_RETRIES` (default 6; 429s wait for the server's retry hint), `<PROVIDER>_REASONING_EFFORT`.
A run where any row fell back to rules is flagged, and does not set the one-time TEST lock.

## v2 test set: many business domains + shared words (new)
`ironman_testset_v2.xlsx` / `.csv`: 200 queries across 12 functions (race operations, registration, finance, IT, HR,
legal, procurement, marketing and sponsorship, medical and safety, facilities, volunteers, travel) plus 20 cross-domain
pairs where one word (ticket, account, policy, transfer, licence, bike ...) means different things.
Visible columns: id, query, label, domain, ambiguous_term, expected_route, expected_domain (no split or difficulty column). 40 per intent.
The DEV/TEST split (70 / 130) and difficulty still exist for the code, in a hidden `_splits` sheet of the xlsx (or `ironman_testset_v2.splits.csv` next to the csv),
merged by `id`, so `--split dev|test` and DEV-only few-shot keep working. The results xlsx also leaves those two columns out.
To clean your own workbook the same way: `python hide_columns.py your.xlsx` (writes `your_clean.xlsx`).
The labels are the author's judgement: review them against your own definitions before relying on the score.

Shared-word handling: the prompt tells the LLM to classify by what is being ASKED (rule/procedure -> DOC_RETRIEVAL,
counts/lists/status -> STRUCTURED_DATA, do something -> UNSUPPORTED, complaint/emergency/human -> ESCALATION,
greeting/thanks -> CONVERSATION), never by the topic word. Few-shot examples (DEV only) are spread across domains
(`FEWSHOT_PER_INTENT`, default 6 per intent for a set with a domain column).
`evaluate.py` now also prints accuracy by domain and on shared-word rows (sheet `by_domain` in the xlsx).

```
python build_testset_v2.py                                        # regenerate the workbook (optional)
export TESTSET_PATH=ironman_testset_v2.xlsx
python evaluate.py --backend rules  --split all                   # offline: ~64% (rules cannot do this set)
python evaluate.py --backend gemini --split dev                   # tune on DEV
python evaluate.py --backend gemini --split test                  # TEST once
```

## v3: domain in the output + repeated questions
**Domain.** Every response now has `domain`: the business function the question is about (Race Operations, Registration and
Athlete Services, Finance, IT, HR, Legal and Compliance, Procurement and Vendors, Marketing and Sponsorship, Medical and Safety,
Facilities and Logistics, Volunteers and Staff Operations, Travel and Accommodation, or General). The LLM returns it together with
the intent (the domain follows the subject, the intent follows the request type); if the LLM is down a keyword fallback fills it in
(`domains.py`). It is not used for routing. `/route` returns it at the top level and inside `classification`.
`ironman_testset_v2` has an `expected_domain` column, and `evaluate.py` prints domain accuracy (informational, not part of the 90% target).

**Repeated questions.** Send an optional `session_id`. The first answer to a question is remembered (`history.py`); asking the same question
again later in that session, even after many other questions, returns that first answer unchanged with `"repeated": true` and no LLM call.
Case, spacing and a trailing `?` do not matter. Without a `session_id` one shared bucket is used. Fallback answers and context-dependent
follow-ups ("explain that simpler") are not remembered. Env: `HISTORY_MAX` (questions per session, default 200), `HISTORY_TTL` (seconds, default 86400).
```
curl -s -X POST localhost:8000/route -H "Content-Type: application/json" \
  -d '{"query":"What is the refund policy?","session_id":"demo"}'
```

## v4: rephrased repeats (no extra tokens)
A reworded repeat in the same session now also returns the first answer (`repeated: true`, plus `matched_question` and `similarity`),
decided locally by `similarity.py`: no LLM call, no API tokens.
1. **Guard (rules):** the two questions must be the same kind of request (count/list, action, human/complaint, greeting) and mention the
   same numbers/IDs, names, time words and negation. This stops lookalikes with different answers from matching
   ("What is the refund policy?" vs "How many refunds were issued?", "athlete 48213" vs "48214", "Dubai" vs "Kona").
2. **Score:** `lexical` (default, no downloads): word overlap after stop-words, stemming and a small synonym table; threshold 0.80.
   `embedding` (optional): a small local model that also catches rewordings with no shared keywords ("how do I get my money back").
   `pip install fastembed` (about 130 MB model downloaded once), then `SIMILARITY_BACKEND=embedding` or `auto`.
Env: `SIMILARITY_BACKEND` (auto|lexical|embedding|off, default auto = embedding if installed else lexical), `SIMILARITY_THRESHOLD`.
Measure and calibrate on your own pairs: `python check_similarity.py [--backend embedding --sweep]` using `paraphrase_pairs.csv`
(24 rephrasings + 23 lookalikes). Keep wrong matches at 0: a wrong match returns the wrong stored answer, a missed one only costs a normal call.

## Status
Done: classifier, router, API, loader, evaluation, error analysis, rate-limit handling, tests (42 passing).
Not yet done: live LLM accuracy. It could not be measured in the build sandbox (outbound access to the LLM
providers was blocked), so run steps 3-4 on your machine. The rules baseline scores (97% dev / 94% held-out on the
generic demo set) were written after seeing that set and only prove the pipeline runs.
